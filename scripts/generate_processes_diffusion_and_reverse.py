import os
import argparse
import numpy as np
import subprocess
import shlex

def generate_workers(commands, log_files):
    workers = []
    for i in range(len(commands)):
        args_list = shlex.split(commands[i])
        stdout = open(log_files[i], "a")
        print('executing %d-th command:\n' % i, args_list)
        p = subprocess.Popen(args_list, stdout=stdout)
        workers.append(p)

    for p in workers:
        p.wait()


model_flags = {}
model_flags['imagenet64'] = "--attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --dropout 0.1 --image_size 64 --learn_sigma True --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --use_new_attention_order True --use_fp16 True --use_scale_shift_norm True"
model_flags['lsun_bedroom'] = "--attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True"
model_flags['lsun_cat'] = model_flags['lsun_bedroom']

model_path_and_guidance = {}
model_path_and_guidance['imagenet64'] = '--classifier_scale 1.0 --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt'
model_path_and_guidance['lsun_bedroom'] = '--model_path models/lsun/lsun_bedroom.pt'
model_path_and_guidance['lsun_cat'] = '--model_path models/lsun/lsun_cat.pt'

execute_file={}
execute_file['imagenet64'] = 'classifier_sample_diffusion_and_reverse.py'
execute_file['lsun_bedroom'] = 'image_sample_diffusion_and_reverse.py'
execute_file['lsun_cat'] = execute_file['lsun_bedroom']


import pdb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--execute', action='store_true', help='whether to execute')
    parser.add_argument('--machine_idx', type=int, default=0, help='the mahcine index')
    parser.add_argument('--num_machines', type=int, default=1, help='num machines to use')
    parser.add_argument('--batch_size', type=int, default=32, help='batch_size to generate images')
    
    parser.add_argument('--reverse_steps', type=int, default=25, help='num of reverse denoising steps')
    parser.add_argument('--chain_length', type=str, default='250', help='the chain length of the accelerated ddpm by jumping steps. It could also be ddim25 is we want to use ddim sampling')

    parser.add_argument("--start_from_scratch", action='store_true', help='whether to generate images purely from scratch, not use gan or vae generated samples. If true, we will run the full chain_length steps')
    parser.add_argument("--num_samples", type=int, default=50000, help='num of samples to generate, only valid when start_from_scratch is true. But when performing sr tasks, this argument will never be used')
    

    parser.add_argument('--dataset', type=str, default='imagenet64', help='the dataset we are generating samples for. This will affect the model flags. Could also be lsun_bedroom, lsun_cat')
    parser.add_argument("--dataset_path", default='../evaluations/precomputed/biggan_deep_imagenet64.npz', type=str, help='path to the gan or vae generated images, we will add noise to them and then use ddpm to denoise')
    parser.add_argument("--save_dir_suffix", default='', type=str, help='save_dir_suffix')

    
    # it only works for imagenet64 dataset
    parser.add_argument("--classifier_guidance_scale", default=1, type=float, help='classifier guidance scale')

    parser.add_argument("--not_eval_metrics", action='store_true', help='whether to not evaluate image generation metrics: fid, sfid, is, precision, recall')

    parser.add_argument('--devices', type=str, default='0,1,2,3,4,5,6,7', help='Cuda visible devices. This script runs one job in parallel. You can specify how many gpus and which gpus to use through this argument. The scipt will automatically split the job to the multple gpus specified.')
    
    args = parser.parse_args()

    model_path_and_guidance['imagenet64'] = f'--classifier_scale {args.classifier_guidance_scale} --classifier_path models/64x64_classifier.pt --classifier_depth 4 --model_path models/64x64_diffusion.pt'


    if args.devices == 'none':
        print('User has not specified a cuda device')
        print('Using the system (Slurm) allocated device', os.environ["CUDA_VISIBLE_DEVICES"])
        devices = os.environ["CUDA_VISIBLE_DEVICES"]
    else:
        devices = args.devices #[0,1,2,3,4,5,6,7]
    
    devices = devices.split(',')
    devices = [int(d) for d in devices]

    num_machines = args.num_machines
    machine_idx = args.machine_idx
    # devices = [0,1,2,3,4,5,6,7]
    ranks = [d + len(devices)*machine_idx for d in range(len(devices))]
    world_size = num_machines * len(devices)

    # total_num_samples = 50000
    # num_samples_per_machine = int(np.ceil(total_num_samples / num_machines))
    # num_samples_per_gpu = int(np.ceil(num_samples_per_machine / 8))

    dataset_path = args.dataset_path # '../evaluations/precomputed/biggan_deep_trunc1_imagenet256.npz'
    diffusion_steps = args.chain_length
    reverse_steps = args.reverse_steps
    dataset = args.dataset
    
    # set up save dir
    if args.start_from_scratch:
        save_dir = 'exps/%s/ddpm_%s_steps/generate_from_scratch' % (dataset, diffusion_steps)
    else:
        save_dir = 'exps/%s/ddpm_%s_steps/diffusion_and_reverse_%d_steps' % (dataset, diffusion_steps, reverse_steps)

    if len(args.save_dir_suffix) > 0:
        save_dir = save_dir + '_' + args.save_dir_suffix
        

    batch_size = args.batch_size

    log_dir = 'logs'
    command = []
    log_files = []
    for i in range(len(devices)):
        if args.start_from_scratch:
            log_file = 'ddpm_%s_length_global_rank_%d.log' % (diffusion_steps, ranks[i])
        else:
            log_file = 'ddpm_%s_length_reverse_%d_steps_global_rank_%d.log' % (diffusion_steps, reverse_steps, ranks[i])
        log_file = os.path.join(log_dir, log_file)

        NEW_FLAGS=f"--global_rank {ranks[i]} --world_size {world_size} --device {devices[i]} --save_dir {save_dir} --save_png_files --save_numpy_array"
        if args.start_from_scratch:
            NEW_FLAGS = NEW_FLAGS + ' --start_from_scratch --num_samples %d' % args.num_samples
        else:
            NEW_FLAGS = NEW_FLAGS + f' --dataset_path {dataset_path} --denoise_steps {reverse_steps}'
                
        SAMPLE_FLAGS=f"--batch_size {batch_size} --timestep_respacing {diffusion_steps}"
        if 'ddim' in diffusion_steps:
            SAMPLE_FLAGS = SAMPLE_FLAGS + ' --use_ddim True'
        MODEL_FLAGS=model_flags[dataset]
        path_and_guidance_flags = model_path_and_guidance[dataset]

        exe_file = execute_file[dataset]

        this_command = f'python {exe_file} {NEW_FLAGS} {MODEL_FLAGS} {path_and_guidance_flags} {SAMPLE_FLAGS}'
        
        print('%d-th command is: %s, stdout to %s\n' % (i, this_command, log_file))
        command.append(this_command)
        log_files.append(log_file)

    if args.execute:
        os.makedirs(log_dir, exist_ok=True)
        generate_workers(command, log_files)

        if machine_idx== 0:
            if not args.not_eval_metrics:
                eval_command = 'python gather_numpyz_and_compute_fid.py --wait_time 60 --dir %s --num_file %d --dataset %s' % (save_dir, world_size, dataset)
                print('executing gather and evaluation command:')
                print(eval_command)
                os.system(eval_command)