import os

from helper.directory_helper import DirectoryHelper

root_path = DirectoryHelper.root_path()

# Config yaml file
config_path = os.path.join(root_path, 'config/experiment.yaml')

# Data of the dataset
dataset_path = os.path.join(root_path, 'data/')