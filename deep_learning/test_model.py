import torch
import torch.nn as nn
#from torch.autograd import Variable
import torch.cuda as cuda

from sklearn.metrics import jaccard_score, f1_score
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from ast import literal_eval

import os
import sys
import pickle
import matplotlib.pyplot as plt
import matplotlib

from learning_classes import dataset, BinaryDiceLoss, U_net, U_net2, U_net3
from utils import print_param_summary, append_scores, load_sample_df, get_dataset_scores, print_progessbar

data_path = '/content/drive/My Drive/BML/DSTL_data/'
mask_path = data_path+'masks/'
img_path = data_path+'processed_img/'
output_path = '/content/drive/My Drive/BML/Ouput_data/'

# the full data
df = load_sample_df(data_path+'train_wkt_v4.csv', class_type=class_type, others_frac=non_class_fraction, seed=1)
