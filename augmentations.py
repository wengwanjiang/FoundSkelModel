import torch.nn.functional as F
import torch
import random
import numpy as np


def joint_courruption(input_data, joint_num=17):                                                                     

    out = input_data.copy()

    flip_prob  = random.random()

    if flip_prob < 0.5:
        joint_indicies = np.random.choice(joint_num, 15,replace=False)
        out[:,:,joint_indicies,:] = 0 
        return out
    
    else:
         joint_indicies = np.random.choice(joint_num, 15,replace=False)
         
         temp = out[:,:,joint_indicies,:] 
         Corruption = np.array([
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] ])
         temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption)
         temp = temp.transpose(3, 0, 1, 2)
         out[:,:,joint_indicies,:] = temp
         return out



def pose_augmentation(input_data):


        Shear       = np.array([
                      [1,	random.uniform(-1, 1), 	random.uniform(-1, 1)],
                      [random.uniform(-1, 1), 1, 	random.uniform(-1, 1)],
                      [random.uniform(-1, 1), 	random.uniform(-1, 1),      1]
                      ])

        temp_data = input_data.copy()
        result =  np.dot(temp_data.transpose([1, 2, 3, 0]),Shear.transpose())
        output = result.transpose(3, 0, 1, 2)

        return output

def temporal_cropresize(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape

    # Temporal crop
    min_crop_length = 64

    scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
    temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

    start = np.random.randint(0, num_of_frames-temporal_crop_length+1)
    temporal_context = input_data[:,start:start+temporal_crop_length, :, :]
    
    # interpolate
    temporal_context = torch.tensor(temporal_context,dtype=torch.float)
    #print(start, temporal_crop_length, scale, num_of_frames)
    try:
        temporal_context=temporal_context.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
    except:
        print(start, temporal_crop_length, scale, num_of_frames)
        exit(0)
    temporal_context=temporal_context[None, :, :, None]
    temporal_context= F.interpolate(temporal_context, size=(output_size, 1), mode='bilinear',align_corners=False)
    temporal_context = temporal_context.squeeze(dim=3).squeeze(dim=0) 
    temporal_context=temporal_context.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

    
    return temporal_context

def crop_subsequence(input_data,num_of_frames,l_ratio,output_size):


    C, T, V, M =input_data.shape

    if l_ratio[0] == 0.1:
    # if training , sample a random crop

         min_crop_length = 64
         scale = np.random.rand(1)*(l_ratio[1]-l_ratio[0])+l_ratio[0]
         temporal_crop_length = np.minimum(np.maximum(int(np.floor(num_of_frames*scale)),min_crop_length),num_of_frames)

         start = np.random.randint(0,num_of_frames-temporal_crop_length+1)
         temporal_crop = input_data[:,start:start+temporal_crop_length, :, :]

         temporal_crop= torch.tensor(temporal_crop,dtype=torch.float)
         temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
         temporal_crop=temporal_crop[None, :, :, None]
         temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
         temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
         temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

         return temporal_crop

    else:
    # if testing , sample a center crop

        start = int((1-l_ratio[0]) * num_of_frames/2)
        data =input_data[:,start:num_of_frames-start, :, :]
        temporal_crop_length = data.shape[1]

        temporal_crop= torch.tensor(data,dtype=torch.float)
        temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
        temporal_crop=temporal_crop[None, :, :, None]
        temporal_crop= F.interpolate(temporal_crop, size=(output_size, 1), mode='bilinear',align_corners=False)
        temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
        temporal_crop=temporal_crop.contiguous().view(C, V, M, output_size).permute(0, 3, 1, 2).contiguous().numpy()

        return temporal_crop
    
def seg2clip(input_data,num_of_frames,l_ratio,output_size, segment, label):
    

    C, T, V, M =input_data.shape
    _, start, end = segment[0], segment[1], segment[2]
    center, dur = (start+end)//2, end-start
    #if dur < 0.5 * output_size:
    bg_ratio = np.random.uniform(0.5, 0.6) 
    bg_frames = int(output_size * bg_ratio)
    action_frames = 64 - bg_frames
    front_bg_frames = int(output_size * np.random.uniform(0,0.5))
    back_bg_frames = bg_frames - front_bg_frames
    front_index = [max(0,i) for i in range(start-front_bg_frames, start)]
    back_index = [min(i,num_of_frames-1) for i in range(end, end+back_bg_frames)]
    front_bg_data = input_data[:, front_index, :, :]
    back_bg_data = input_data[:, back_index, :, :]
    assert len(front_index) + len(back_index) == bg_frames

    if dur <= action_frames + 10:
        data =input_data
    else:
        left = np.random.randint(start, (center-action_frames*0.5)-1)#start + action_frames * 0.5
        right = np.random.randint(center+action_frames*0.5+1, end)
        data =input_data[:,left:right, :, :]
    
    temporal_crop_length = data.shape[1]
    temporal_crop= torch.tensor(data,dtype=torch.float)
    temporal_crop=temporal_crop.permute(0, 2, 3, 1).contiguous().view(C * V * M,temporal_crop_length)
    temporal_crop=temporal_crop[None, :, :, None]
    temporal_crop= F.interpolate(temporal_crop, size=(action_frames, 1), mode='bilinear',align_corners=False)
    temporal_crop=temporal_crop.squeeze(dim=3).squeeze(dim=0) 
    temporal_crop=temporal_crop.contiguous().view(C, V, M, action_frames).permute(0, 3, 1, 2).contiguous().numpy()
    

    front_bg_label = label[front_index]
    mid_action_label = np.array([segment[0] for idx in range(action_frames)])
    back_bg_label = label[back_index]
    label = np.concatenate([front_bg_label, mid_action_label, back_bg_label])

    temporal_crop = np.concatenate([front_bg_data, temporal_crop, back_bg_data], axis=1)
    assert temporal_crop.shape[1] == 64 and label.shape[0] == 64

    return temporal_crop, label
