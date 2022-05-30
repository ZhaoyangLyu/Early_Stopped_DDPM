import numpy as np
import argparse
import os
import time
import shutil

def find_file_in_a_dir(directory, key_word):
    files = os.listdir(directory)
    for f in files:
        if key_word in f:
            return f
    return False

def check_all_file_generated(directory, total_ranks):
    for rank in range(total_ranks):
        filename = False
        while not filename:
            filename = find_file_in_a_dir(directory, 'rank_%d.npz' % rank)
            if not filename:
                print('waiting rank_%d.npz to be generated' % rank)
                time.sleep(args.wait_time) # wait 5 min
    return True

if __name__ == '__main__':
    ref_batch = {}
    ref_batch['imagenet64'] = 'precomputed/VIRTUAL_imagenet64_labeled.npz'
    ref_batch['lsun_bedroom'] = 'precomputed/lsun/VIRTUAL_lsun_bedroom256.npz'
    ref_batch['lsun_cat'] = 'precomputed/lsun/VIRTUAL_lsun_cat256.npz'

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='exps/ddpm_250_steps', help='the directory where the npz files are stored')
    parser.add_argument('--num_file', type=int, default=8, help='num of files to gather')
    parser.add_argument('--wait_time', type=int, default=300, help='waiting time to generate npz file')
    parser.add_argument('--dataset', type=str, default='imagenet64', help='the dataset we are generating samples for. This will affect the ref batch. Could also be lsun_bedroom, lsun_cat')
    
    args = parser.parse_args()

    result_dir = args.dir #'exps/ddpm_250_steps'
    total_ranks = args.num_file #32
    saved_samples = 50000
    images = []
    labels = []

    if find_file_in_a_dir(result_dir, 'samples_all.npz'):
        print('files have already been gathered to the file samples_all.npz')
    else:
        check_all_file_generated(result_dir, total_ranks)
        for rank in range(total_ranks):
            filename = find_file_in_a_dir(result_dir, 'rank_%d.npz' % rank)
            print('loading file', filename)
            data = np.load(os.path.join(result_dir, filename))
            images.append(data['arr_0'])
            if 'arr_1' in data.files:
                labels.append(data['arr_1'])
            data.close()

        images = np.concatenate(images)
        print('gathered images are of shape', images.shape)

        save_labels = False
        if len(labels) > 0:
            labels = np.concatenate(labels)
            print('gathered labels are of shape', labels.shape)
            save_labels = True

        images = images[0:saved_samples]
        print('saved images are of shape', images.shape)
        save_file = os.path.join(result_dir, 'samples_all.npz')
        if save_labels:
            labels = labels[0:saved_samples]
            print('saved labels are of shape', labels.shape)
            np.savez(save_file, images, labels)
        else:
            np.savez(save_file, images)

    
    image_folder = find_file_in_a_dir(result_dir, 'images')
    if image_folder:
        image_folder = os.path.join(result_dir, image_folder)
        print('removing the image folder', image_folder)
        shutil.rmtree(image_folder)
        # files = os.listdir(image_folder)
        # print('There are %d images in the folder %s' % (len(files), image_folder))
    for rank in range(total_ranks):
        filename = find_file_in_a_dir(result_dir, 'rank_%d.npz' % rank)
        if filename:
            filename = os.path.join(result_dir, filename)
            print('removing the file', filename)
            os.remove(filename)

    os.chdir('../evaluations')
    directory_name = os.path.split(result_dir)[-1]
    job_name = 'fid_' + directory_name.split('_')[-2]

    this_ref_batch = ref_batch[args.dataset]
    command = f'python evaluator.py {this_ref_batch} ../scripts/{result_dir}/samples_all.npz'
    print('executing command', command)
    try:
        os.system(command)
        # print('removing', f'../scripts/{result_dir}/samples_all.npz')
        # os.remove(f'../scripts/{result_dir}/samples_all.npz')
    except Exception as e:
        print('Some error happens during evaluation')
        print('error message is:\n', e)
