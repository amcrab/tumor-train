#!/usr/bin/env python3

## This script converts paired HDF5 files to image files and annotation CSV.
## Originally designed for PCam dataset downloaded here: https://github.com/basveeling/pcam
## Author: Angela Crabtree

######################## LIBRARIES ########################

import pandas as pd
import os
import h5py
import numpy as np
import gzip
from PIL import Image
import argparse

######################## ARGPARSE ########################

# options
parser = argparse.ArgumentParser(description='Convert paired HDF5 files to image files and annotation file.')
parser.add_argument('-o', '--output', type=str, 
    help='output directory (default = image file directory)')

# required args
requiredargs = parser.add_argument_group('required arguments')
requiredargs.add_argument('-x', '--img_file', type=str, 
    help='HDF5 image file', required=True)
requiredargs.add_argument('-y', '--labels_file', type=str, 
    help='HDF5 image file', required=True)

# parse arguments
args = parser.parse_args()

# assign optional variables
if args.output == None:
    fdir = os.path.dirname(args.img_file)
else: 
    fdir = args.output

#################### labels #####################

# open labels hdf5 file
fpath = args.labels_file
with gzip.open(fpath, 'r') as gzf: # open zipped file
    dataset = h5py.File(gzf, 'r')

    # Print all root level object names (aka keys) 
    all_keys = list(dataset.keys())
    print(f'\n\tKeys in this file: {list(all_keys)}')
    k = all_keys[0] # get first object name/key
    print("\n\tShape of 1st key array:", dataset[k].shape)

    # convert data to pandas array
    y = np.array(dataset[k]).flatten()
    df = pd.DataFrame({'label': y})

    # save dataframe to file
    outfile=os.path.basename(fpath).split(".h")[0]+".csv"
    labels_csv = os.path.join(fdir, outfile)
    df.to_csv(labels_csv, index=False)
    print(f'\n\tCSV saved as: {outfile}\n')

    # close labels hdf5 file
    dataset.close()

#################### images #####################

# open image hdf5 file
fpath = args.img_file
with gzip.open(fpath, 'r') as gzf: # open zipped file
    print("Opening " + os.path.basename(fpath) + "...")
    dataset = h5py.File(gzf, 'r')

    # Print all root level object names (aka keys) 
    all_keys = list(dataset.keys())
    print(f'\n\tKeys in this file: {list(all_keys)}')
    k = all_keys[0] # get first object name/key
    print("\n\tShape of 1st key array:", dataset[k].shape)

    # convert data to jpeg images and save filenames to dict
    n_obs, img_h, img_w, img_color = dataset[k].shape
    fname_dict={} # keep dict of image names
    base_img_name=os.path.basename(fpath).split(".h")[0].split("_")[4]
    for i in range(n_obs):
        img_np = np.array(dataset[k])[i,:,:,:]
        im = Image.fromarray(img_np)
        fname = "pcam_"+base_img_name+"_"+str(i)+".jpg" 
        print(f'\n\tSaving image {i+1} of {n_obs}')
        im.save(os.path.join(fdir, fname))
        fname_dict[i]=os.path.join(fdir, fname)

    # save annotation file
    df = pd.DataFrame(fname_dict.values(), columns=['file'])
    df = df.append(pd.read_csv(labels_csv), ignore_index=True)
    outfile="pcam_annot.csv"
    df.to_csv(os.path.join(fdir, outfile), index=False)
    print(f'\n\tCSV saved as: {outfile}\n')

    # close hdf5 file
    dataset.close()