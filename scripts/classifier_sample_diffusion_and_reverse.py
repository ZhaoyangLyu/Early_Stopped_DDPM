"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F

# from guided_diffusion import dist_util, logger
from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

from save_image_utils import save_images
from npz_dataset import NpzDataset, DummyDataset
from imagenet_dataloader.imagenet_dataset import ImageFolderDataset

import pdb

def get_dataset(path, global_rank, world_size):
    if os.path.isfile(path): # samples could be store in a .npz file
        dataset = NpzDataset(path, rank=global_rank, world_size=world_size)
    else:
        dataset = ImageFolderDataset(path, label_file='./imagenet_dataloader/imagenet_val_labels.pkl', transform=None, 
                        permute=True, normalize=True, rank=global_rank, world_size=world_size)
    return dataset

def main():
    args = create_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    device = th.device('cuda')
    save_dir = args.save_dir if len(args.save_dir)>0 else None

    # dist_util.setup_dist()
    logger.configure(dir = save_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # model.load_state_dict(
    #     dist_util.load_state_dict(args.model_path, map_location="cpu")
    # )
    model.load_state_dict(
        th.load(args.model_path, map_location="cpu")
    )
    # model.to(dist_util.dev())
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #     dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    # )
    # classifier.to(dist_util.dev())
    classifier.load_state_dict(
        th.load(args.classifier_path, map_location="cpu")
    )
    classifier.to(device)
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def general_cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            loss = selected.sum() * args.classifier_scale
            return th.autograd.grad(loss, x_in)[0] 

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("loading dataset...")
    # load gan or vae generated images
    if args.start_from_scratch:
        dataset = DummyDataset(args.num_samples, rank=args.global_rank, world_size=args.world_size)
    else:
        dataset = get_dataset(args.dataset_path, args.global_rank, args.world_size)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)


    if args.save_png_files:
        os.makedirs(os.path.join(logger.get_dir(), 'images'), exist_ok=True)
        start_idx = args.global_rank * dataset.num_samples_per_rank

    logger.log("sampling...")
    all_images = []
    all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    for i, data in enumerate(dataloader):
        image, label = data
        cond_fn = lambda x,t,y : general_cond_fn(x, t, y=y)
        if args.start_from_scratch:
            shape = (image.shape[0], 3, args.image_size, args.image_size)
        else:
            shape = list(image.shape)
        if args.start_from_scratch:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(shape[0],), device=device)
        else:
            classes = label.to(device).long()
        image = image.to(device)
        model_kwargs = {}
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.start_from_scratch:
            sample = sample_fn(
                model_fn,
                shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device
            )
        else:
            sample = sample_fn(
                model_fn,
                shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=device,
                noise=image,
                denoise_steps=args.denoise_steps
            )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        
        sample = sample.detach().cpu().numpy()
        classes = classes.detach().cpu().numpy()
        if args.save_png_files:
            save_images(sample, classes, start_idx + len(all_images) * args.batch_size, os.path.join(logger.get_dir(), 'images'))
        all_images.append(sample)
        all_labels.append(classes)
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    if args.save_numpy_array:
        arr = np.concatenate(all_images, axis=0)
        # arr = arr[: args.num_samples]
        label_arr = np.concatenate(all_labels, axis=0)
        # label_arr = label_arr[: args.num_samples]

        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_rank_{args.global_rank}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    # add arguments
    parser.add_argument("--device", default=0, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--global_rank", default=0, type=int, help='global rank of this process')
    parser.add_argument("--world_size", default=1, type=int, help='the total number of ranks')
    parser.add_argument("--save_dir", default='', type=str, help='the directory to save the generate images')
    parser.add_argument("--save_png_files", action='store_true', help='whether to save the generate images into individual png files')
    parser.add_argument("--save_numpy_array", action='store_true', help='whether to save the generate images into a single numpy array')
    
    # these two arguments are only valid when not start from scratch
    parser.add_argument("--denoise_steps", default=25, type=int, help='number of denoise steps')
    parser.add_argument("--dataset_path", default='../evaluations/precomputed/biggan_deep_imagenet64.npz', type=str, help='path to the generated images. Could be an npz file or an image folder')
    
    parser.add_argument("--start_from_scratch", action='store_true', help='whether to generate images purely from scratch, not use gan or vae generated samples')
    # num_samples is defined elsewhere, num_samples is only valid when start_from_scratch
    # parser.add_argument("--num_samples", type=int, default=50000, help='num of samples to generate, only valid when start_from_scratch is true')
    return parser

import pdb
if __name__ == "__main__":
    main()
