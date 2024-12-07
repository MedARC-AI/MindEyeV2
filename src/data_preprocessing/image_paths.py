'''
This script prepares image data for model training.

It creates list of imagefiles
    preprocessed_eeg_training[0,:,:,:] will correspond to the image /00001_aardvark/aardvark_01b.jpg, 
    preprocessed_eeg_training[1,:,:,:] will correspond to the image /00001_aardvark/aardvark_02s.jpg 
    etc...

Before running this script make sure you have downloaded images from THINGSEEG2 dataset.
See data_preprocessing/download_data.py

Usage:
    python image_paths.py --data_path <absolute path to data directory>
'''

import argparse
import numpy as np
import os


img_train_dir = 'Images/training_images'
img_test_dir = 'Images/test_images'
get_data_dir = 'GetData'


def img_prep(data_absolute_path):
    #Image list

    training_imgpaths = np.empty(0, dtype = '<U10') # classes x num of images per class
    test_imgpaths = np.empty(0, dtype = '<U10') 
    
    for dirpath, dirnames, filenames in os.walk(os.path.join(data_absolute_path,img_train_dir)):
        for filename in filenames:
            # Construct the full path to the file
            full_path = os.path.join(data_absolute_path, dirpath, filename)
            training_imgpaths = np.append(training_imgpaths, full_path)

    print('\t Train images processed')
    for dirpath, dirnames, filenames in os.walk(os.path.join(data_absolute_path,img_test_dir)):
        for filename in filenames:
            # Construct the full path to the file
            full_path = os.path.join(data_absolute_path,dirpath, filename)
            test_imgpaths = np.append(test_imgpaths, full_path)
    print('\t Test images processed')
    print('Saving...')
    np.save(os.path.join(data_absolute_path, get_data_dir,'test_imgpaths.npy'), test_imgpaths)
    np.save(os.path.join(data_absolute_path, get_data_dir,'training_imgpaths.npy'), training_imgpaths)

            


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='data absolute path')
    args = parser.parse_args()

    print("IMAGE PREP STARTED")
    img_prep(args.data_path)
    print("SUCCESS")
   
    

