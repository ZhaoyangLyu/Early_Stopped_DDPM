"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

# from guided_diffusion import dist_util, logger
from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from save_image_utils import save_images
from npz_dataset import NpzDataset, DummyDataset
from imagenet_dataloader.imagenet_dataset import ImageFolderDataset

def get_dataset(path, global_rank, world_size, class_cond):
    if os.path.isfile(path): # base_samples could be store in a .npz file
        dataset = NpzDataset(path, rank=global_rank, world_size=world_size)
    else:
        # if class cond, we assume the dataset is imagenet
        label_file = './imagenet_dataloader/imagenet_val_labels.pkl' if class_cond else 0
        dataset = ImageFolderDataset(path, label_file=label_file, transform=None, 
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

    logger.log("loading dataset...")
    if args.start_from_scratch:
        dataset = DummyDataset(args.num_samples, rank=args.global_rank, world_size=args.world_size)
    else:
        # dataset = NpzDataset(args.dataset_path, rank=args.global_rank, world_size=args.world_size)
        dataset = get_dataset(args.dataset_path, args.global_rank, args.world_size, args.class_cond)
    dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)

    if args.save_png_files:
        os.makedirs(os.path.join(logger.get_dir(), 'images'), exist_ok=True)
        start_idx = args.global_rank * dataset.num_samples_per_rank

    logger.log("sampling...")
    all_images = []
    all_labels = []
    # while len(all_images) * args.batch_size < args.num_samples:
    for i, (image, label) in enumerate(dataloader):
        shape = list(image.shape)
        classes = label.to(device).long()
        image = image.to(device)
        model_kwargs = {}
        if args.class_cond:
            if args.start_from_scratch:
                classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device)
            model_kwargs["y"] = classes
        sample_fn = (diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop)
        if args.start_from_scratch:
            sample = sample_fn(
                model,
                (image.shape[0], 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs
            )
        else:
            sample = sample_fn(
                model,
                shape,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
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
        if args.class_cond:
            all_labels.append(classes)
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    if args.save_numpy_array:
        arr = np.concatenate(all_images, axis=0)
        # arr = arr[: args.num_samples]
        if args.class_cond:
            label_arr = np.concatenate(all_labels, axis=0)
        # label_arr = label_arr[: args.num_samples]

        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}_rank_{args.global_rank}.npz")
        logger.log(f"saving to {out_path}")
        if args.class_cond:
            np.savez(out_path, arr, label_arr)
        else:
            np.savez(out_path, arr)

    # dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    
    # add arguments
    parser.add_argument("--device", default=0, type=int, help='the cuda device to use to generate images')
    parser.add_argument("--global_rank", default=0, type=int, help='global rank of this process')
    parser.add_argument("--world_size", default=1, type=int, help='the total number of ranks')
    parser.add_argument("--save_dir", default='', type=str, help='the directory to save the generate images')
    parser.add_argument("--save_png_files", action='store_true', help='whether to save the generate images into individual png files')
    parser.add_argument("--save_numpy_array", action='store_true', help='whether to save the generate images into a single numpy array')
    
    parser.add_argument("--denoise_steps", default=25, type=int, help='number of denoise steps')
    parser.add_argument("--dataset_path", default='../evaluations/precomputed/biggan_deep_trunc1_imagenet256.npz', type=str, help='path to the generated images')

    parser.add_argument("--start_from_scratch", action='store_true', help='whether to generate images purely from scratch, not use gan or vae generated samples')
    # parser.add_argument("--num_samples", type=int, default=50000, help='num of samples to generate, only valid when start_from_scratch is true')
    
    return parser

import pdb
if __name__ == "__main__":
    main()
