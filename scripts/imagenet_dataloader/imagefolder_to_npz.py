from tqdm import tqdm
import numpy as np
import torch
import os
import shutil
import argparse

from imagenet_dataset import ImageFolderDataset

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import pdb
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=str, default='./', help='the root folder')
    args = parser.parse_args()
    root = args.folder

    dataset = ImageFolderDataset(root, transform=None, permute=False, normalize=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=8)

    imgs = []
    labels = []

    with torch.no_grad():

        for i, data in enumerate(tqdm(dataloader)):
            x, y = data
            imgs.append(x)
            labels.append(y)

        
        imgs = torch.cat(imgs).numpy()
        labels = torch.cat(labels).numpy()

        print('Images are of shape', imgs.shape, imgs.dtype)
        print('labels are of shape', labels.shape, labels.dtype)

        folder, name = os.path.split(root)
        name = name + '.npz'
        print('Saving npz dataset to', os.path.join(folder, name))
        np.savez(os.path.join(folder, name), imgs, labels)
    
            