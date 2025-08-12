# sys
import pickle

# torch
import torch
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
np.set_printoptions(threshold=np.inf)

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
        self.label = []
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
            self.label.append(item['label'])
            self.number_of_frames.append(conf.shape[1])
        self.number_of_frames = np.array(self.number_of_frames)
        self.label = np.array(self.label, dtype=np.int32)
    def __len__(self):
        return int(self.N) # * self.semi)
 
    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        number_of_frames = self.number_of_frames[index]
        label = self.label[index]

        # crop a sub-sequnce 
        data_numpy = augmentations.crop_subsequence(data_numpy, number_of_frames, self.l_ratio, self.input_size)

        # return data_numpy, label
          
        # joint representation
        jt = data_numpy.transpose(1,3,2,0)
        jt = jt.reshape(self.input_size,self.M*self.V*self.C).astype('float32')
        js = data_numpy.transpose(3,2,1,0)
        js = js.reshape(self.M*self.V, self.input_size*self.C).astype('float32')
        # bone representation
        bone = np.zeros_like(data_numpy)
        for v1,v2 in self.Bone:
            bone[:,:,v1-1,:] = data_numpy[:,:,v1-1,:] - data_numpy[:,:,v2-1,:]
        bt = bone.transpose(1,3,2,0)
        bt = bt.reshape(self.input_size,self.M*self.V*self.C).astype('float32')
        bs = bone.transpose(3,2,1,0)
        bs = bs.reshape(self.M*self.V, self.input_size*self.C).astype('float32')

        # motion representation
        motion = np.zeros_like(data_numpy) 
        motion[:,:-1,:,:] = data_numpy[:,1:,:,:] - data_numpy[:,:-1,:,:]  
        mt = motion.transpose(1,3,2,0)
        mt = mt.reshape(self.input_size,self.M*self.V*self.C).astype('float32')
        ms = motion.transpose(3,2,1,0)
        ms = ms.reshape(self.M*self.V, self.input_size*self.C).astype('float32')

        return jt, js, bt, bs, mt, ms, label
        
      
