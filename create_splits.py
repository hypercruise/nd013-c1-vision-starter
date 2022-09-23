import argparse
import glob
import os
import random
import shutil
# import numpy as np

from utils import get_module_logger

def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    
    # Read all file names in the source folder
    datafiles = [filename for filename in glob.glob(f'{source}/*.tfrecord')]
    print(len(datafiles))

    # Create destination subfolders if the datafiles are not empty
    subfolders_list = ['train', 'val', 'test']

    if (len(datafiles) > 0):
        for subpath in subfolders_list:
            dest_path = os.path.join(destination, subpath)
            os.makedirs(dest_path, exist_ok=True)

    # Randomize the file name list
    random.shuffle(datafiles)

    # Calculate the numbers of files for train, val, and test with the ratio of 0.8:0.1:0.1
    total_files = len(datafiles)
    train_num = int(0.8 * total_files)
    val_num = int(0.1 * total_files)
    # test_num = int(0.1 * total_files)     # Not used

    # Copy to the train, val, and test folders
    count = 0
    for file in datafiles:
        count += 1

        if (count <= train_num):
            idx_folder = 0      # train subfolder
        elif (count <= train_num + val_num):
            idx_folder = 1      # val subfolder
        else:
            idx_folder = 2      # test subfolder

        dest_path=os.path.join(destination, subfolders_list[idx_folder])
        dest_path=os.path.join(dest_path,os.path.basename(file))
        print('Copy: ', count, dest_path)
        shutil.copy(file, dest_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)