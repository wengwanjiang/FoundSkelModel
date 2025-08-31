import time
import torch, pickle

import numpy as np
#np.set_printoptions(threshold=np.inf)
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
                 l_ratio,
                 input_size):
        
        self.input_size=input_size
        self.crop_resize =True
        self.l_ratio = l_ratio
        self.load_data(data_path)
        self.N, self.C, self.T, self.V, self.M = len(self.data), 3, 64, 25, 2
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        
        
        print(len(self.data),len(self.number_of_frames),len(self.label))
        print("l_ratio",self.l_ratio)
        #self.use_txt = use_txt
    def load_data(self, data_path):
        # data: N C V T M
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)
        # load num of valid frame length
        self.number_of_frames= data_dict['num_frames']
        self.data = data_dict['data']
        self.label = data_dict['label']

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        #return np.random.rand(3,512,25,2), np.random.rand(3,512,25,2), np.random.rand(3,512,25,2), np.random.rand(3,512,25,2), np.random.rand(3,512,25,2)
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]
        
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