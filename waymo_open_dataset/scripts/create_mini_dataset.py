'''
This script will sample mini-dataset from the dataset_original folder and save it dataset_mini folder
to use it, just rename dataset_mini to dataset
Args: num-segments: nuber of segments to be added to mini-dataset, every segment contains 8 ~ 10 frames 
'''

import os, argparse
import random
from pathlib import Path
import subprocess
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument('num_segments', type=int, default=10, help='number of segments in the mini dataset')
FLAGS = parser.parse_args()

# locate original dataset
ORIGINAL_DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset_original')
if not os.path.exists(ORIGINAL_DATA_DIR):
    raise ValueError('cannot lcoate the original dataset, make sure that folder dataset_original exist')

# create directory for mini dataset
MINI_DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset_mini')
MINI_TRAIN_DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset_mini', 'train')
MINI_TRAIN_VOTES_DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset_mini', 'train', 'votes')
MINI_VAL_DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset_mini', 'val')
MINI_VAL_VOTES_DATA_DIR = os.path.join(BASE_DIR, '..', 'dataset_mini', 'val', 'votes')

# folder hierarchy will be established correctly if you create the deepest folders only !
Path(MINI_TRAIN_VOTES_DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(MINI_VAL_VOTES_DATA_DIR).mkdir(parents=True, exist_ok=True)

# handle train data

# get list of all segments in the original dataset
original_train_dir = os.path.join(ORIGINAL_DATA_DIR, 'train')
train_segments_list = [s for s in os.listdir(original_train_dir) if s.split('-')[0] == 'segment']
print("len of training segments are {}".format(len(train_segments_list)))
train_sampling = random.choices(train_segments_list, k=FLAGS.num_segments)
print('\n'.join(train_sampling))

for segment in train_sampling:
    # copy segment folder
    segment_dir = os.path.join(original_train_dir, segment)
    new_segment_dir = os.path.join(MINI_TRAIN_DATA_DIR, segment)
    if not os.path.exists(segment_dir):
        raise ValueError('check path concatenations')
    p2 = subprocess.run('cp -r {} {}'.format(segment_dir, new_segment_dir), shell=True)
     
    # copy segment votes folder
    segment_votes_dir = os.path.join(original_train_dir, 'votes', segment)
    new_segment_votes_dir = os.path.join(MINI_TRAIN_VOTES_DATA_DIR, segment)
    if not os.path.exists(segment_votes_dir):
        raise ValueError('check path concatenations')
    p2 = subprocess.run('cp -r {} {}'.format(segment_votes_dir, new_segment_votes_dir), shell=True)
    
   




# handle validation data, copy the entire validation folder, only five segments

original_val_dir = os.path.join(ORIGINAL_DATA_DIR, 'val')
val_segments_list = [s for s in os.listdir(original_val_dir) if s.split('-')[0] == 'segment']
print("len of validation segments are {}".format(len(val_segments_list)))
p2 = subprocess.run('cp -r {} {}'.format(original_val_dir, MINI_DATA_DIR), shell=True)


