root_path = "/data/wwj_bkup/UmURL0711/data/"
root_path = "./data/data/"
root_path = "/data/wwj/"
# model arguments
encoder_arguments = {
   "t_input_size":150,
   "s_input_size":192,
   "hidden_size":1024,
   "num_head":1,
   "num_layer":2,
   "alpha" : 0.5,
   "kernel_size": 1,
   "gap": 4
 }





class  opts_uav_cross_subject():
  def __init__(self, ver):
   self.name = f'uavXsub_{ver}'
   self.encoder_args = encoder_arguments
   self.encoder_args["t_input_size"] = 102 # 150 // 25 * 17
   # feeder
   self.train_feeder_args = {
     "data_path": f"/home/wengwanjiang/AAAI25/USDRL/UAVHuman_For_TE-GCN-main/data/{ver}/train_data.npy",
     "num_frame_path": f"/home/wengwanjiang/AAAI25/USDRL/UAVHuman_For_TE-GCN-main/data/{ver}/train_num_of_frames.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }



   
   
class  opts_ntu_60_cross_subject_2d():

  def __init__(self, padding):
   self.name = 'ntu60Xsub_2d'
   self.encoder_args = encoder_arguments
   self.encoder_args["t_input_size"] = 102 # 150 // 25 * 17
   # feeder
   self.train_feeder_args = {
     "pkl_path": root_path + "ntu60_hrnet.pkl",
     "split": "train", 
     "protocol": "xsub",
     "padding": padding,
     "l_ratio": [0.1,1],
     "input_size": 64
   }

 
class  opts_ntu_120_cross_subject_2d():

  def __init__(self, padding='confidence'):
   self.name = 'ntu120Xsub_2d'
   self.encoder_args = encoder_arguments
   self.encoder_args["t_input_size"] = 102 # 150 // 25 * 17
   # feeder
   self.train_feeder_args = {
     "pkl_path": root_path + "ntu120_hrnet.pkl",
     "split": "train", 
     "protocol": "xsub",
     "padding": padding,
     "l_ratio": [0.1,1],
     "input_size": 64
   }



class  opts_ntu_120_cross_setup_2d():

  def __init__(self, padding='confidence'):
   self.name = 'ntu120Xsetup_2d'
   self.encoder_args = encoder_arguments
   self.encoder_args["t_input_size"] = 102 # 150 // 25 * 17
   # feeder
   self.train_feeder_args = {
     "pkl_path": root_path + "ntu120_hrnet.pkl",
     "split": "train", 
     "protocol": "xset",
     "padding": padding,
     "l_ratio": [0.1,1],
     "input_size": 64
   }



class  opts_ntu_60_cross_view_2d():

  def __init__(self, padding='confidence'):
   self.name = 'ntu60Xview_2d'
   self.encoder_args = encoder_arguments
   self.encoder_args["t_input_size"] = 102 # 150 // 25 * 17
   # feeder
   self.train_feeder_args = {
     "pkl_path": root_path + "ntu60_hrnet.pkl",
     "split": "train", 
     "protocol": "xview",
     "padding": padding,
     "l_ratio": [0.1,1],
     "input_size": 64
   }

class opts_pku_v2_xsub():

  def __init__(self):
   self.name = 'pkuv2Xsub'
   self.encoder_args = encoder_arguments

   # feeder
   self.train_feeder_args = {
     "data_path": root_path + "/PKUMMD/PKUv2_xsub_train.pkl",
     "l_ratio": [0.1,1],
     "input_size": 64
   }
class  opts_ntu_60_cross_view():

  def __init__(self):
   self.name = 'ntu60Xview'
   self.encoder_args = encoder_arguments

   # feeder
   self.train_feeder_args = {
     "data_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_data_joint.npy",
     "num_frame_path": root_path + "/NTU-RGB-D-60-AGCN/xview/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }

class  opts_ntu_60_cross_subject():

  def __init__(self):
   self.name = 'ntu60Xsub'
   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": root_path + "/NTU-RGB-D-60-AGCN/xsub/train_data_joint.npy",
     "num_frame_path": root_path + "/NTU-RGB-D-60-AGCN/xsub/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }

class  opts_ntu_120_cross_subject():

  def __init__(self):
   self.name = 'ntu120Xsub'
   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": root_path + "/NTU-RGB-D-120-AGCN/xsub/train_data_joint.npy",
     "num_frame_path": root_path + "/NTU-RGB-D-120-AGCN/xsub/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }

class  opts_ntu_120_cross_setup():

  def __init__(self):
   self.name = 'ntu120Xset'
   self.encoder_args = encoder_arguments
   
   # feeder
   self.train_feeder_args = {
     "data_path": root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_data_joint.npy",
     "num_frame_path": root_path + "/NTU-RGB-D-120-AGCN/xsetup/train_num_frame.npy",
     "l_ratio": [0.1,1],
     "input_size": 64
   }

