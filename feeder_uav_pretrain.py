import time
import torch

import numpy as np
import random

try:
    from feeder import augmentations
except:
    import augmentations


class Feeder(torch.utils.data.Dataset):
    """ 
    Arguments:
        data_path: the path to '.npy' data, the shape of data should be (N, C, T, V, M)
    """

    def __init__(self,
                 data_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 mmap=True):
        
        self.input_size=input_size
        self.crop_resize =True
        self.l_ratio = l_ratio
        self.load_data(data_path=data_path, label_path=None, num_of_frames_path=num_frame_path)
        # self.load_data(mmap)

        self.N, self.C, self.T, self.V, self.M = len(self.data), 3, 64, 17, 2 #self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 4), (3,5), (4,6), (5,7), (7,9), (8,10), (9, 11), (10, 8), (11, 9), (12, 14), (13,15), (14, 16), (15, 17), (16,14), (17,15)]
        
        
        print(len(self.data),len(self.number_of_frames))
        print("l_ratio",self.l_ratio)
    def load_data(self, data_path, num_of_frames_path, label_path):
        self.data = np.load(data_path)
        # def normalize_skeleton_data(data):
        #     xyz_data = data[:, :3, :, :, :]  # shape=(N, 3, T, V, M)
        #     mean = np.mean(xyz_data, axis=(0, 2, 3, 4), keepdims=True)  # shape=(1, 3, 1, 1, 1)
        #     std = np.std(xyz_data, axis=(0, 2, 3, 4), keepdims=True)    # shape=(1, 3, 1, 1, 1)
        #     std = np.where(std < 1e-7, 1.0, std)
        #     normalized_xyz = (xyz_data - mean) / std
            
        #     normalized_data = normalized_xyz
            
        #     return normalized_data
        # self.data = normalize_skeleton_data(self.data)
        self.number_of_frames = np.load(num_of_frames_path)
    

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        
        number_of_frames = min(self.number_of_frames[index], data_numpy.shape[1])
     
        # apply spatio-temporal augmentations to generate  view 1 
        # temporal crop-resize

        data_numpy_v1_crop = augmentations.temporal_cropresize(data_numpy, number_of_frames, self.l_ratio, self.input_size)
        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v1 = augmentations.joint_courruption(data_numpy_v1_crop)
        else:
                 data_numpy_v1 = augmentations.pose_augmentation(data_numpy_v1_crop)


        # apply spatio-temporal augmentations to generate  view 2
        # temporal crop-resize
        data_numpy_v2_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)
        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v2 = augmentations.joint_courruption(data_numpy_v2_crop)
        else:
                 data_numpy_v2 = augmentations.pose_augmentation(data_numpy_v2_crop)
                 
                 
        # apply spatio-temporal augmentations to generate  view 3
        # temporal crop-resize
        data_numpy_v3_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)
        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v3 = augmentations.joint_courruption(data_numpy_v3_crop)
        else:
                 data_numpy_v3 = augmentations.pose_augmentation(data_numpy_v3_crop)
                 
                 
        # apply spatio-temporal augmentations to generate  view 4
        # temporal crop-resize
        data_numpy_v4_crop = augmentations.temporal_cropresize(data_numpy,number_of_frames, self.l_ratio, self.input_size)
        # randomly select  one of the spatial augmentations 
        flip_prob  = random.random()
        if flip_prob < 0.5:
                 data_numpy_v4 = augmentations.joint_courruption(data_numpy_v4_crop)
        else:
                 data_numpy_v4 = augmentations.pose_augmentation(data_numpy_v4_crop)
        
        return data_numpy_v1, data_numpy_v2, data_numpy_v3, data_numpy_v4

        