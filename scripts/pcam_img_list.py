#!/usr/bin/env python3

## This script generates a list of image files for use in Nightingale workflow. 
## Authors: Angela Crabtree

######################## LIBRARIES ########################

import argparse
import os
import pandas as pd
import re

######################## ARGPARSE ########################

# options
parser = argparse.ArgumentParser(description='Generate list tile annotations of PCam data.')
parser.add_argument('-o', '--output', type=str, 
    help='output file (default = "tile_annot.csv")')
parser.add_argument('-n', '--num_imgs', type=int, 
    help='limit number of random images to process from input directory')

# required args
requiredargs = parser.add_argument_group('required arguments')
requiredargs.add_argument('-i', '--image_dir', type=str, 
    help='Directory containing tile files', required=True)
requiredargs.add_argument('-L', '--labels_file', type=str, 
    help='CSV file containing labels', required=True)

# parse arguments and save to args variables
args = parser.parse_args()

# assign args to variables, if necessary
if args.output == None:
    output_dir = args.image_dir[0]
    outfile = os.path.join(output_dir, "/tile_annot.csv")
else: 
    outfile = args.output

######################## FUNCTIONS ########################

def get_tile_file_list(tile_dir: str) -> list:
    '''return list of filenames from all tiles within folder.'''
    img_file_list=[]
    for item in os.listdir(tile_dir):
        # only append list if file is .jpg file
        if os.path.basename(item).split(".")[-1]=="jpg": 
            f = os.path.join(tile_dir, item)
            img_file_list.append(f)
    return img_file_list

def get_tile_labels(img_file_list: list, y_pd: pd.DataFrame) -> pd.DataFrame:
    slide_info_dict = {}
    for i in range(len(img_file_list)):
        f = img_file_list[i] # filename
        tile_id = os.path.basename(f).split(".jpg")[0] 
        tile_num = int(tile_id.split("_")[2])
        label = y_pd.loc[tile_num, 'label']
        slide_info_dict[i]=[f, label]
    # save as dataframe
    idx=["file","label"]
    slide_info_pd = pd.DataFrame.from_dict(slide_info_dict, orient='index', columns=idx)
    return slide_info_pd

######################## MAIN ########################

## Load tile filenames and labels
labels_df = pd.read_csv(args.labels_file) # assumes header is "label"
tile_file_list = get_tile_file_list(args.image_dir)

## subset image list if user limits number to use
if args.num_imgs != None:
    tile_file_list = sample(tile_file_list, args.num_imgs)

## return all required slide info in one data frame
tile_info_df = get_tile_labels(tile_file_list, labels_df)
#print(tile_info_df.head()) # check dataframe structure

## save dataframe to csv file
tile_info_df.to_csv(outfile, index=False)