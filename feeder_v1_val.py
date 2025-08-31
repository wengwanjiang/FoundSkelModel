import pickle
import torch
import numpy as np

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
                 input_size):#, use_txt=False):
        print('v1 val', data_path)
        self.input_size=input_size
        self.l_ratio = l_ratio


        self.load_data(data_path)
        self.N, self.C, self.T, self.V, self.M = len(self.data_dict), 3, 64, 25, 2
        self.S = self.V
        self.B = self.V
        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        
        
        print(len(self.data_dict))
        print("l_ratio",self.l_ratio)
        #self.use_txt = use_txt
    def load_data(self, data_path):
        # data: N C V T M
        with open(data_path, 'rb') as f:
            self.data_dict = pickle.load(f)

    def __len__(self):
        return self.N

    def __iter__(self):
        return self

    def __getitem__(self, index):

        # get raw input

        # input: C, T, V, M
        sample_name = self.data_dict[index]['name']
        data_numpy = np.array(self.data_dict[index]['data'], dtype=np.float32)
        label = self.data_dict[index]['label']

        # random chooose an action clip
        
        
       
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

        return jt, js, bt, bs, mt, ms, label, sample_name