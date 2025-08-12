import time
import torch

import numpy as np
import random
import pickle
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
                 split, protocol, padding,
                 pkl_path,
                 l_ratio,
                 input_size,):
        assert split in ['train', 'val']
        assert padding in ['confidence', 'zero']
        assert protocol in ['xsub', 'xview', 'xset']
        self.split = split 
        self.pkl_path = pkl_path
        self.input_size=input_size
        
        self.padding = padding
        self.crop_resize =True
        self.l_ratio = l_ratio
        self.load_pkl(pkl_path=pkl_path, padding=padding, protocol=protocol, split=split)
        self.protocol = protocol 
        
        self.N, self.C, self.T, self.V, self.M = len(self.data), 3, 64, 17, 2 #self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 4), (3,5), (4,6), (5,7), (7,9), (8,10), (9, 11), (10, 8), (11, 9), (12, 14), (13,15), (14, 16), (15, 17), (16,14), (17,15)]
        
        
        print(len(self.data),len(self.number_of_frames))
        print("l_ratio",self.l_ratio)
    def load_pkl(self, pkl_path, padding, protocol, split):
        
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        _split = set(data['split'][f'{protocol}_{split}'])

        self.data = []
        self.number_of_frames = []
        for item in data['annotations']:
            file_name = item['frame_dir']
            if file_name not in _split:
                continue
            height, width = item['img_shape']
            
            keypoint = np.array(item['keypoint'])
            conf = np.array(item['keypoint_score'])
            keypoint[:, :, 0] /= height
            keypoint[:, :, 1] /= width

            conf = conf[:, :, :, np.newaxis]
            if padding == 'confidence':
                motion = np.concatenate([keypoint, conf], axis=3)# M T V C
            elif padding == 'zero':
                motion = np.concatenate([keypoint, np.zeros_like(conf, dtype=np.float32)], axis=3)# M T V C
                 
            motion = motion.transpose([3, 1, 2, 0])
            if motion.shape[-1] == 1:
                zero = np.zeros_like(motion)
                motion = np.concatenate([motion, zero], axis=3)
            self.data.append(motion)
            self.number_of_frames.append(conf.shape[1])
        self.number_of_frames = np.array(self.number_of_frames)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):
        
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

        