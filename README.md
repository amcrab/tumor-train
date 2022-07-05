# tumor-train
ML model for determining whether H&amp;E tile image contains tumor tissue

# Data

Download data from [https://github.com/basveeling/pcam](https://github.com/basveeling/pcam)

Or, if  using a linux environment: 

'''
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_test_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_train_y.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_meta.csv
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_x.h5.gz
wget https://zenodo.org/record/2546921/files/camelyonpatch_level_2_split_valid_y.h5.gz
'''

# Environment

        conda create --name pcam --file ./env/environment.yml

# Usage and Tips

