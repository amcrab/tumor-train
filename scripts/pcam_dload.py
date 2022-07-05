#!/usr/bin/env python3

## This script creates a pytorch DataLoader object from PCam dataset
## Author: Angela Crabtree
## with code modified from: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html

######################## LIBRARIES ########################

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split

######################## ARGPARSE ########################

# options
parser = argparse.ArgumentParser(description='Create DataLoader object from tiles.')
parser.add_argument('-i', '--img_dir', type=str, 
    help='Folder where images are stored, if different from annotations file')
parser.add_argument('-o', '--outfile', type=str, 
    help='output filename')
    
# required args
requiredargs = parser.add_argument_group('required arguments')
requiredargs.add_argument('-a', '--annotations', type=str, 
    help='CSV containing filenames in column 1 and labels in column 2', required=True)

# call parser
args = parser.parse_args()

# assign args to optional variables
if args.outfile == None:
    output_dir = os.path.dirname(args.annotations)
    outfile = os.path.join(output_dir, pcam_dataloaders.pth)
else: 
    output_dir = os.path.dirname(args.outfile)
    outfile = args.outfile

######################## FUNCTIONS ########################

class PCamDataset(Dataset):
    """PCam tumor tile dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.annot_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
    
    # return number of observations (images/labels)
    def __len__(self):
        return len(self.annot_df)

    # images are not stored in the memory at once but read as required with __getitem__
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # define where images will be found
        img_path = self.annot_df.iloc[idx, 0]
        if self.root_dir != None: # if user specifies args.img_dir
            img_path = os.path.join(
                self.root_dir, os.path.basename(self.annot_df.iloc[idx, 0]))
        image = Image.open(img_path) # open image
        tumor_label = self.annot_df.iloc[idx, 1]
        tumor_label = np.array([tumor_label], dtype=float)
        tumor_label = tumor_label.astype('float').reshape(-1)
        sample = {'image': image, 'label': tumor_label}

        if self.transform:
            sample['image'] = self.transform(sample['image'])

        return sample
    
    def check_imgs(self, outfile):
        fig = plt.figure()
        for i in range(4):
            sample = self[i]
            #print(i, sample['image'].shape, sample['label'].shape)
            ax = plt.subplot(1, 4, i + 1)
            plt.tight_layout()
            ax.set_title('Sample #{}'.format(i))
            ax.axis('off')
            img = sample['image']
            if isinstance(img, torch.Tensor): # permute img if tensor
                img = img.permute(1,2,0).numpy()
            plt.imshow(img)
            plt.title(sample['label'], fontweight ="bold")
        plt.savefig(outfile)

def get_train_valid_loader(
    dataset, batch_size=16, split=True, num_workers=4, val_ratio=0.15, pin_memory=True):
    train, val = train_test_split(dataset, test_size=val_ratio)
    #train_aug=augmentation(train,'H')
    train_loader = DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(
        val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


######################## MAIN ########################

if __name__ == "__main__":

    # instantiate untransformed data
    pcam_dataset = PCamDataset(
        csv_file=args.annotations,
        root_dir=args.img_dir)
    # check that images and labels match up (checks first 4 images/labels)
    pcam_dataset.check_imgs(os.path.join(
        os.path.dirname(args.annotations), "check_tiles.jpg"))

    # instantiate transformed data
    transformed_dataset = PCamDataset(
        csv_file=args.annotations,
        root_dir=args.img_dir,
        transform=transforms.Compose([
            transforms.Resize(56), # enter vgg16 at 3rd module
            transforms.ToTensor()
            ])
        )
    # check that images and labels match up (checks first 4 images/labels)
    transformed_dataset.check_imgs(os.path.join(
        os.path.dirname(args.annotations), "check_tiles_trans.jpg"))

    # create DataLoader objects from dataset
    dataloaders = get_train_valid_loader(transformed_dataset, batch_size=64)

    # save DataLoaders
    torch.save(dataloaders, outfile)