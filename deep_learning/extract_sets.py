import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation
import numpy as np
import pandas as pd
import skimage
import tifffile
import pickle
import os
import rasterio

from utils import get_crops_grid, load_image_part, get_represented_classes, get_samples

data_path = '/content/drive/My Drive/BML/DSTL_data/'
mask_path = data_path+'masks/'
img_path = data_path+'processed_img/'
output_path = '/content/drive/My Drive/BML/Ouput_data/'

order_dict = {1:'building', 2:'misc', 3:'road', 4:'track', \
              5:'tree', 6:'crop', 7:'water', 8:'vehicle'}

# %% load the img_id
img_id_list =  list(pd.read_csv(data_path+'train_wkt_v4.csv', usecols=['ImageId']).ImageId.unique())

# define which image is for test and which is for train
id_test = ['6100_2_2', '6060_2_3', '6110_4_0', '6160_2_1']
id_train = [id for id in img_id_list if id not in id_test]

#%% Generate samples
crop_size_train = (160,160)
overlap_train = (80,80)
class_offset_train = (40,40)
class_area_train = (80,80)

crop_size_test = (1024,1024)
overlap_test = None
class_offset_test = (0,0)
class_area_test = crop_size_test

df_train = get_samples(id_train, \
                       img_path, mask_path, \
                       crop_size_train, overlap_train, \
                       order_dict, \
                       class_offset_train, class_area_train, \
                       verbose=True)

df_test = get_samples(id_test, \
                       img_path, mask_path, \
                       crop_size_test, overlap_test, \
                       order_dict, \
                       class_offset_test, class_area_test, \
                       verbose=True)

# the test set cut similarly to the train for performance estimations
df_test_small = get_samples(id_test, \
                       img_path, mask_path, \
                       crop_size_train, overlap_train, \
                       order_dict, \
                       class_offset_train, class_area_train, \
                       verbose=True)

df_train.to_csv(data_path+'train_wkt_v4.csv')
df_test.to_csv(data_path+'test_samples.csv')
df_test_small.to_csv(data_path+'test_samples_small.csv')

# -----------------------------------------------------------------------------------------
#%%
subimg = load_image_part((0,0), (5*80,8*80), img_path+id_train[2]+'.tif')
crops = get_crops_grid(subimg.shape[1], subimg.shape[2], crop_size_train, overlap_train)
#%%
fig, ax = plt.subplots(1,1,figsize=(10,7))
#fig.patch.set_alpha(0)
ax.imshow(np.moveaxis(subimg[[4,2,1], :, :], 0, 2))
ax.set_axis_off()
ax.set_title('train sample generation example', fontsize=12)
P1 = ax.add_patch(matplotlib.patches.Rectangle((0, 0), 0, 0, fc=(0,0,0,0), ec='Orangered', lw=2))
P2 = ax.add_patch(matplotlib.patches.Rectangle((0, 0), 0, 0, fc=(0.1,0.1,0.1,0.3), ec='dodgerblue', lw=1))
fig.legend([P1, P2], ['train sample', 'loss-relevant area'], ncol=2, loc='lower center', fontsize=12)
fig.tight_layout()

def init():
    return []

def animate(c):
    P1.set_xy((c[1], c[0]))
    P1.set_height(crop_size_train[0])
    P1.set_width(crop_size_train[1])
    #P2.set_xy((c[1]+overlap_train[0]/2, c[0]+overlap_train[1]/2))
    #P2.set_height(overlap_train[0])
    #P2.set_width(overlap_train[1])
    P2 = ax.add_patch(matplotlib.patches.Rectangle((c[1]+overlap_train[0]/2, c[0]+overlap_train[1]/2), overlap_train[0], overlap_train[1], fc=(0.1,0.1,0.1,0.5), ec='dodgerblue', lw=1))
    return [P1,P2]

anim = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=crops, interval=500, blit=True)
anim.save(output_path+'/Figures/train_samples.gif', writer='imagemagick', dpi=150)
