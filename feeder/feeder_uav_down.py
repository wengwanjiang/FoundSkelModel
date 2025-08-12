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
                 data_path,
                 label_path,
                 num_frame_path,
                 l_ratio,
                 input_size,
                 mmap=True):#, use_txt=False):

        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
     
        self.l_ratio = l_ratio


        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 4), (3,5), (4,6), (5,7), (7,9), (8,10), (9, 11), (10, 8), (11, 9), (12, 14), (13,15), (14, 16), (15, 17), (16,14), (17,15)]
        if 'train' in data_path:
            self.semi = 1
        else:
            self.semi = 1
        N = self.N
        print('ori',self.data.shape,len(self.number_of_frames),len(self.label), self.semi)
        print("l_ratio",self.l_ratio)
        #self.use_txt = use_txt
        # idx = np.arange(N)
        # np.random.shuffle(idx)
        # #np.random.shuffle(idx)
        # N_used = int(N * self.semi)
        # idx_used = idx[: N_used]
        # self.data = self.data[idx_used]
        # self.label = np.array(self.label)[idx_used]
        # self.number_of_frames = self.number_of_frames[idx_used]
        # print('used', self.data.shape,len(self.number_of_frames),len(self.label))
    def load_data(self, mmap):
        # data: N C V T M

        # load data
        self.data = np.load(self.data_path)
        
        # def normalize_skeleton_data(data):
        #     xyz_data = data[:, :3, :, :, :]  # shape=(N, 3, T, V, M)
        #     mean = np.mean(xyz_data, axis=(0, 2, 3, 4), keepdims=True)  # shape=(1, 3, 1, 1, 1)
        #     std = np.std(xyz_data, axis=(0, 2, 3, 4), keepdims=True)    # shape=(1, 3, 1, 1, 1)
        #     std = np.where(std < 1e-7, 1.0, std)
        #     normalized_xyz = (xyz_data - mean) / std
            
        #     normalized_data = normalized_xyz
            
        #     return normalized_data
        # self.data = normalize_skeleton_data(self.data)
        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)


        # load label
        if '.pkl' in self.label_path:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)

        

    def __len__(self):
        return int(self.N * self.semi)
 
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
        
      
