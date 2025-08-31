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
                 mmap=True,observe_ratio=1):

        self.data_path = data_path
        self.label_path = label_path
        self.num_frame_path= num_frame_path
        self.input_size=input_size
     
        self.l_ratio = l_ratio


        self.load_data(mmap)
        self.N, self.C, self.T, self.V, self.M = self.data.shape
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        if 'train' in data_path:
            self.semi = 1
        else:
            self.semi = 1
        N = self.N
        print('origin ',self.data.shape,len(self.number_of_frames),len(self.label), self.semi)
        print(f"Observe Ratio = {self.observe_ratio} , l_ratio = {self.l_ratio}")

        idx = np.arange(N)
        np.random.shuffle(idx)
        #np.random.shuffle(idx)
        N_used = int(N * self.semi)
        idx_used = idx[: N_used]
        self.data = self.data[idx_used]
        self.label = np.array(self.label)[idx_used]
        self.number_of_frames = self.number_of_frames[idx_used]
        print('used ', self.data.shape,len(self.number_of_frames),len(self.label))
        self.observe_ratio = observe_ratio
    def load_data(self, mmap):
        # data: N C V T M

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        # load num of valid frame length
        self.number_of_frames= np.load(self.num_frame_path)

        # load label
        if '.pkl' in self.label_path:
            with open(self.label_path, 'rb') as f:
                self.sample_name, self.label = pickle.load(f)
        elif '.npy' in self.label_path:
                self.label = np.load(self.label_path).tolist()
        

    def __len__(self):
        return int(self.N * self.semi)
 
    def __iter__(self):
        return self

    def __getitem__(self, index):
        
        # get raw input

        # input: C, T, V, M
        data_numpy = np.array(self.data[index])
        # number_of_frames = self.number_of_frames[index]
        number_of_frames = max(3,(self.number_of_frames[index] * self.observe_ratio) // 1)
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
        
      
