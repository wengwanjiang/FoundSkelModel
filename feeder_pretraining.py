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
                 mmap=True, observe_ratio=1):

        self.data_path = data_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
        self.crop_resize =True
        self.l_ratio = l_ratio
        
        self.load_data(mmap)

        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        

        print(self.data.shape,len(self.number_of_frames))
        self.observe_ratio = observe_ratio
        print(f"Observe Ratio = {self.observe_ratio} , l_ratio = {self.l_ratio}")
        
        # number_of_frames = max(3,(self.number_of_frames[index] * self.observe_ratio) // 1)
    def load_data(self, mmap):
        # data: N C T V M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        
        # number_of_frames = self.number_of_frames[index]
        number_of_frames = max(3,(self.number_of_frames[index] * self.observe_ratio) // 1)
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

        