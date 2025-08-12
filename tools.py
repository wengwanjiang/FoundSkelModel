import numpy as np
import os, sys, shutil
import pickle
import yaml, torch
from datetime import datetime
from easydict import EasyDict as edict
from typing import Any, IO
def sum_para_cnt(model):
    return sum([param.nelement() for param in model.parameters()])

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    def get_str(self):
        formatted_num = "{:.4f}".format(self.avg)
        return self.name+': ' + str(formatted_num) + '\t'

def remove_prefix(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.split('module.')[-1]
        new_state_dict[k] = v 
    return new_state_dict
            
