'''
Downloads data from THINGSEEG2 dataset raw data

Images
EEG - train and test sets for 10 subjects

Full datased size ~=50GB 

USAGE:
    python download_data.py --data_path <absolute path to data directory>

If you only want to download data for specified subjects:
    python download_data.py --data_path <absolute path to data directory> --subject-indices 1 2 3 

    If you only want to download images
    python download_data.py --data_path <absolute path to data directory> --no_eeg
'''
import os
import zipfile
import urllib.request
import argparse

get_data_dir = 'GetData'
preprocessed_eeg_dir = 'PreprocessedEEG'
raw_eeg_dir = 'RawEEG'
images_dir = 'Images'



def setup_dir(dir_path):
    if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")
    else:
        print(f"Directory already exists: {dir_path}")



def download_data(url, filename):
    # Download the file
    try:
        urllib.request.urlretrieve(url, filename)  # Change filename as needed
        print(f'Download of {filename} completed successfully.')
    except Exception as e:
        print(f'Failed to download {filename}. Error: {e}')


def unzip_data(zip_file_path, zip_directory='.'):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(zip_directory)  # Extract to the same directory
        print(f'Extracted: {zip_file_path} into {zip_directory}')

    # Delete the zip file
    os.remove(zip_file_path)
    print(f'Deleted zip file: {zip_file_path}')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, help='data absolute path')
    parser.add_argument('--subject_indices', nargs='+', type=int, default=[i for i in range (1,11)])
    parser.add_argument('--test_imgs_only', action="store_true")
    parser.add_argument('--no_eeg', action="store_true", default=False)
    args = parser.parse_args()

    data_absolute_path = args.data_path
    os.makedirs(data_absolute_path, exist_ok=True)

    sub_ids = args.subject_indices
    test_imgs_only = args.test_imgs_only
    no_eeg = args.no_eeg

    get_data_dir = os.path.join( data_absolute_path, get_data_dir)
    preprocessed_eeg_dir = os.path.join(data_absolute_path, preprocessed_eeg_dir)
    raw_eeg_dir = os.path.join(data_absolute_path, raw_eeg_dir)
    images_dir = os.path.join(data_absolute_path, images_dir)

    # Define files to download from OSF
    metadata = [
        {  
            'dir' : images_dir,
            'filename' : 'test_images.zip',
            'url': 'https://osf.io/download/znu7b/' 
        },
        {  
            'dir' : images_dir,
            'filename' : 'training_images.zip',
            'url': 'https://osf.io/download/3v527/' 
        },
        {  
            'dir' : get_data_dir,
            'filename' : 'image_metadata.npy',
            'url': 'https://osf.io/download/qkgtf/'
        },
        {  
            'dir' : raw_eeg_dir,
            'filename' : 'osfstorage-archive.zip',
            'url': 'https://plus.figshare.com/ndownloader/articles/18470912/versions/4'
        }

    ]

    if no_eeg:
        metadata.pop(3) #remove eeg for all subjects (zip) metadata
    if test_imgs_only:
        metadata.pop(1) #remove training images metadata
   

    setup_dir(get_data_dir) 
    setup_dir(preprocessed_eeg_dir)
    setup_dir(images_dir)
    setup_dir(raw_eeg_dir)

    #download and unzip the data
    for entry in metadata:
        download_data(entry['url'], os.path.join(entry['dir'], entry['filename']))
        if (entry['filename']).endswith('.zip'):
           unzip_data(os.path.join(entry['dir'], entry['filename']), entry['dir'])  

    # unzip data for all subjects
    if not no_eeg:
        for sub_id in range(1,11):
            filename = f'sub-{str(sub_id).zfill(2)}.zip'
            unzip_data(os.path.join(raw_eeg_dir, filename), raw_eeg_dir)  
    


    
    
  