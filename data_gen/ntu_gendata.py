import argparse
import pickle
from tqdm import tqdm
import sys
from numpy.lib.format import open_memmap

sys.path.extend(['../'])
from preprocess import pre_normalization

# ntu 60
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]

# ntu 120
training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38, 
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 70, 74, 78,80, 81, 82, 
    83, 84, 85, 86, 89, 91, 92, 93, 94, 95, 97, 98, 100, 103
]
training_setups = [ 2, 4,  6, 8, 10, 12, 14,  16,  18,  20, 22, 24, 26, 28, 30, 32]

training_cameras = [2, 3]
max_body_true = 2
max_body_kinect = 4
num_joint = 25
max_frame = 300

import numpy as np
import os


def read_skeleton_filter(file):
    with open(file, 'r') as f:
        skeleton_sequence = {}
        skeleton_sequence['numFrame'] = int(f.readline())
        skeleton_sequence['frameInfo'] = []
        # num_body = 0
        for t in range(skeleton_sequence['numFrame']):
            frame_info = {}
            frame_info['numBody'] = int(f.readline())
            frame_info['bodyInfo'] = []

            for m in range(frame_info['numBody']):
                body_info = {}
                body_info_key = [
                    'bodyID', 'clipedEdges', 'handLeftConfidence',
                    'handLeftState', 'handRightConfidence', 'handRightState',
                    'isResticted', 'leanX', 'leanY', 'trackingState'
                ]
                body_info = {
                    k: float(v)
                    for k, v in zip(body_info_key, f.readline().split())
                }
                body_info['numJoint'] = int(f.readline())
                body_info['jointInfo'] = []
                for v in range(body_info['numJoint']):
                    joint_info_key = [
                        'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 'colorY',
                        'orientationW', 'orientationX', 'orientationY',
                        'orientationZ', 'trackingState'
                    ]
                    joint_info = {
                        k: float(v)
                        for k, v in zip(joint_info_key, f.readline().split())
                    }
                    body_info['jointInfo'].append(joint_info)
                frame_info['bodyInfo'].append(body_info)
            skeleton_sequence['frameInfo'].append(frame_info)

    return skeleton_sequence


def get_nonzero_std(s):  # tvc
    index = s.sum(-1).sum(-1) != 0  # select valid frames
    s = s[index]
    if len(s) != 0:
        s = s[:, :, 0].std() + s[:, :, 1].std() + s[:, :, 2].std()  # three channels
    else:
        s = 0
    return s


def read_xyz(file, max_body=4, num_joint=25):  # 取了前两个body
    seq_info = read_skeleton_filter(file)
    data = np.zeros((max_body, seq_info['numFrame'], num_joint, 3))
    for n, f in enumerate(seq_info['frameInfo']):
        for m, b in enumerate(f['bodyInfo']):
            for j, v in enumerate(b['jointInfo']):
                if m < max_body and j < num_joint:
                    data[m, n, j, :] = [v['x'], v['y'], v['z']]
                else:
                    pass

    # select two max energy body
    energy = np.array([get_nonzero_std(x) for x in data])
    index = energy.argsort()[::-1][0:max_body_true]
    data = data[index]

    data = data.transpose(3, 1, 2, 0)
    return data


def gendata(data_path, out_path, ignored_sample_path=None, benchmark='xview', part='eval'):
    Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
    
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])
        setup_id = int(
            filename[filename.find('S') + 1:filename.find('S') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        elif benchmark == 'xsetup':
            istraining = (setup_id in training_setups)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)

    fl = open_memmap(
        '{}/{}_num_frame.npy'.format(out_path, part),
        dtype='int',
        mode='w+',
        shape=(len(sample_label),))

    fp = np.zeros((len(sample_label), 3, max_frame, num_joint, max_body_true), dtype=np.float32)

    for i, s in enumerate(tqdm(sample_name)):
        data = read_xyz(os.path.join(data_path, s), max_body=max_body_kinect, num_joint=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data
        fl[i] = data.shape[1] # num_frame
    
    motion = np.zeros_like(fp)
    motion[:, :, :-1, :, :] = fp[:, :, 1:, :, :] - fp[:, :, :-1, :, :]
    bone = np.zeros_like(fp)
    for v1, v2 in Bone:
        bone[:, :, :, v1 - 1, :] = fp[:, :, :, v1 - 1, :] - fp[:, :, :, v2 - 1, :]
    
    fp = pre_normalization(fp)
    np.save('{}/{}_data_joint.npy'.format(out_path, part), fp)
    np.save('{}/{}_data_motion.npy'.format(out_path, part), motion)
    np.save('{}/{}_data_bone.npy'.format(out_path, part), bone)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    
    # parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    # parser.add_argument('--data_path', default='../data/nturgbd_raw/nturgb+d_skeletons/')
    # parser.add_argument('--ignored_sample_path',
    #                     default='../data/nturgbd_raw/samples_with_missing_skeletons.txt')
    # parser.add_argument('--out_folder', default='../data/NTU-RGB-D-60-AGCN/')
    # benchmark = ['xsub', 'xview']

    parser.add_argument('--data_path', default='../data/nturgbd_raw_120/nturgb+d_skeletons/')
    parser.add_argument('--ignored_sample_path',
                       default='../data/nturgbd_raw_120/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/NTU-RGB-D-120-AGCN/')
    benchmark = ['xsub','xsetup', ]

    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            print(b, p)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)