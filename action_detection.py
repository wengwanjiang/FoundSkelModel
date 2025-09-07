import argparse
import torch.nn.functional as F
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import numpy as np
from tools import AverageMeter, remove_prefix, sum_para_cnt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# change for action recogniton
from dataset import get_finetune_training_set,get_finetune_validation_set


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('--epochs', default=150, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('-b', '--batch-size', default=256, type=int,metavar='N')

parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

parser.add_argument('--schedule', default=[120, 140,], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')

parser.add_argument('--evaluate', default='false', type=str,
                    help='path to detection checkpoint')

parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')

parser.add_argument('--protocol', default='cross_view', type=str,
                    help='training protocol of ntu')
parser.add_argument('--moda', default='joint', type=str,
                    help='joint, motion , bone')

parser.add_argument('--backbone', default='DSTE', type=str,
                    help='DSTE or STTR')

args = parser.parse_args()
best_acc1 = 0

def load_pretrained(pretrained, model):
    if os.path.isfile(pretrained):
        checkpoint = torch.load(pretrained, map_location="cpu")
        state_dict = checkpoint['state_dict']
        state_dict = remove_prefix(state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
        print("message",msg)
        print(set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))

def load_detector(detector, model):
    if os.path.isfile(detector):
        checkpoint = torch.load(detector, map_location="cpu")
        state_dict = checkpoint['state_dict']
        state_dict = remove_prefix(state_dict)
        msg = model.load_state_dict(state_dict, strict=True)
        print("message",msg)
        assert len(msg.missing_keys) + len(msg.unexpected_keys) == 0, "Not all keys matched successfully."
        print("=> loaded detector '{}'".format(detector))
    else:
        print("=> no checkpoint found at '{}'".format(detector))

def main():
    args = parser.parse_args()
    
    if 'pth' not in args.evaluate and not os.path.exists(args.pretrained):
        print(args.pretrained, ' not found!')
        
    print(type(args.evaluate),args.evaluate)
    if args.evaluate is not None:
        generate_bbox()
    main_worker(args)


def main_worker(args):
    global best_acc1

    # training dataset
    from options  import options_downstream as options 
    if args.finetune_dataset == 'pku_v1':
        opts = options.opts_pku_v1_xsub()
    elif args.finetune_dataset == 'pku_v2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_v2_xsub()
    

    if args.backbone == 'DSTE':
        from model.DSTE import Downstream
        model = Downstream(**opts.encoder_args)
    elif args.backbone == 'STTR':
        from model.STTR import Downstream
        model = Downstream(**opts.encoder_args)

    print(sum_para_cnt(model)/1e6)
    print("options",opts.encoder_args,opts.train_feeder_args,opts.test_feeder_args)
    print('\n', args)


    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()

    # load from pre-trained model
    if args.pretrained and 'pth' not in args.evaluate:
        load_pretrained(args.pretrained, model)
    
        model = nn.DataParallel(model)
        
    model = model.cuda()
    
    
    criterion = nn.CrossEntropyLoss().cuda()

    fc_parameters = []
    other_parameters = []

    for name, param in model.named_parameters():
        if name.startswith('module.fc'):
            param.requires_grad = True
            fc_parameters.append(param)
        else:
            param.requires_grad = True
            other_parameters.append(param)
    fc_parameters = list(fc_parameters)
    other_parameters = list(other_parameters)

    params = [{'params': fc_parameters, 'lr': args.lr}, {'params': other_parameters, 'lr': args.lr}]
    optimizer = torch.optim.SGD(params,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for parm in optimizer.param_groups:
        print ("optimize parameters lr ",parm['lr'])



    train_dataset = get_finetune_training_set(opts)
    val_dataset = get_finetune_validation_set(opts)
    
    trainloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
    }
    valloader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
    }
    train_loader = torch.utils.data.DataLoader(train_dataset,  **trainloader_params)
    val_loader = torch.utils.data.DataLoader(val_dataset,  **valloader_params)
    
    for epoch in range(0, args.epochs):
        # train for one epoch
        
        if args.evaluate is not None:
            detector_path = './checkpoint/ntu60_xs_j_a5b5_sttr/' + args.evaluate
            load_detector(detector_path, model)
            with torch.no_grad():
                generate_bbox(val_loader, model, args)
            break
        train(train_loader, model, criterion, optimizer, epoch, args)
        
        # evaluate on validation set
        if (epoch+1) % 5 == 0:
            state = {'state_dict': model.state_dict()}
            torch.save(state, './checkpoint/ntu60_xs_j_a5b5_sttr/' + str(epoch) + '_detection.pth.tar')
            
            acc1 = validate(val_loader, model, criterion, args)
        else:
            acc1 = 0
        
        # remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # if is_best and False:
        #     print("found new best accuracy:= ",acc1)
        #     best_acc1 = max(acc1, best_acc1)
        #     best_state = {'state_dict': model.state_dict()}
        #     torch.save(best_state, './checkpoint/ntu60_xs_j_a5b5_sttr/best_detection.pth.tar')
    print("class head Final best accuracy",best_acc1)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':1.4f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    bgtop1 = AverageMeter('bgAcc@1', ':3.2f')
    actop1 = AverageMeter('acAcc@1', ':3.2f')    
    top5 = AverageMeter('Acc@5', ':3.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, top1, bgtop1, actop1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()

    end = time.time()
    for i, (jt, js, bt, bs, mt, ms, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True)
        target = target.long().cuda(non_blocking=True)
        # compute output
        output = model(jt, js, bt, bs, mt, ms, knn_eval=False, detect=True)
        #print(output.shape, target.shape)
        output = output.reshape(-1, 52)
        target = target.reshape(-1)
        
        loss = criterion(output, target)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5), ignore=-1)
        bgacc1, _ = accuracy(output, target, topk=(1, 5), ignore=1)
        acacc1, _ = accuracy(output, target, topk=(1, 5), ignore=0)
        losses.update(loss.item(), output.shape[0])
        top1.update(acc1[0], output.shape[0])
        top5.update(acc5[0], output.shape[0])
        bgtop1.update(bgacc1[0], output.shape[0])
        actop1.update(acacc1[0], output.shape[0])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i + 1 == len(train_loader):
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':1.4f')
    top1 = AverageMeter('Acc@1', ':3.2f')
    bgtop1 = AverageMeter('bgAcc@1', ':3.2f')
    actop1 = AverageMeter('acAcc@1', ':3.2f')
    top5 = AverageMeter('Acc@5', ':3.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, bgtop1, actop1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (jt, js, bt, bs, mt, ms, target, sample_name) in enumerate(val_loader):
            
            jt = jt.float().cuda(non_blocking=True)
            js = js.float().cuda(non_blocking=True)
            bt = bt.float().cuda(non_blocking=True)
            bs = bs.float().cuda(non_blocking=True)
            mt = mt.float().cuda(non_blocking=True)
            ms = ms.float().cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            # compute output
            output = model(jt, js, bt, bs, mt, ms, knn_eval=False, detect=True)
            output = output.reshape(-1, 52)
            target = target.reshape(-1)
            loss = criterion(output, target)
            
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5), ignore=-1)
            bgacc1, _ = accuracy(output, target, topk=(1, 5), ignore=1) 
            acacc1, _ = accuracy(output, target, topk=(1, 5), ignore=0)
            losses.update(loss.item(), output.shape[0])
            top1.update(acc1[0], output.shape[0])
            top5.update(acc5[0], output.shape[0])
            bgtop1.update(bgacc1[0], output.shape[0])
            actop1.update(acacc1[0], output.shape[0])
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i + 1 == len(val_loader):
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} * bgAcc@1 {bgtop1.avg:.3f} * acAcc@1 {actop1.avg:.3f}\t\tAcc@5 {top5.avg:.3f}'
              .format(top1=top1, bgtop1=bgtop1, actop1=actop1, top5=top5))
        
    
    return top1.avg

def generate_bbox(val_loader, model, args, thereshold=0.02):
    from tqdm import tqdm
    
    # switch to evaluate mode
    model.eval()
    proposal = {}
    os.mkdir('./checkpoint/ntu60_xs_j_a5b5_sttr/detect_each_frame')
    for i, (jt, js, bt, bs, mt, ms, target, sample_name) in tqdm(enumerate(val_loader)):
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True)
        target = target.long().cuda()
        # compute output
        output = model(jt, js, bt, bs, mt, ms, modality=args,detect=True)# 512, 64, 52
        output = F.softmax(output, dim=-1)  # output = [512, 64, 52]
        sp = './checkpoint/ntu60_xs_j_a5b5_sttr/detect_each_frame/'
        os.makedirs(sp, exist_ok=True)
        
        for idx, file in enumerate(sample_name):
            proposal[file] = []
            with open(os.path.join(sp, file), 'a') as f:
                pred_bs, gt_bs = output[idx], target[idx] # [64, 52], [64]
                pred_fs = torch.argmax(pred_bs, dim=1)
                results = torch.cat((pred_fs.unsqueeze(1), gt_bs.unsqueeze(1), pred_bs), dim=1)
                for result in results:
                    s = ','.join(map(str, result.tolist())) + '\n'
                    f.write(s)
        #exit(0)
    print('========== mask matrix thereshold =', thereshold)
    for file in tqdm(os.listdir(sp)):
        with open(sp + file, 'r') as f:
            data = [u.lstrip().rstrip().split(',') for u in f.readlines()] # pred, gt, score0, ..., score51
            data = np.array(data, dtype=np.float32) # pred, gt, score0, ..., score51 shape = [T, (2 + 52)]
            pb_matrix = data[:, 2:].T   # [52, T]
            mask_matrix = (pb_matrix > thereshold).astype(int) # [52, T] 01matrix
        for i in range(1, mask_matrix.shape[0]): # exclude bg (0)
            pro_ = get_proposal(mask_matrix[i])
            proposal[file] += [[i, u, v, np.mean(pb_matrix[i][u:v])] for u, v in pro_]      
    # temporal_thd = 0.1
    # print('========== tmeporal NMS thereshold =', temporal_thd)
    os.mkdir('./checkpoint/ntu60_xs_j_a5b5_sttr/detect_result')
    for k, v in proposal.items():
        with open('./checkpoint/ntu60_xs_j_a5b5_sttr/detect_result/' + k, 'a') as ff:
            s = '' 
            # for lb, st, ed, score in temporal_nms(v, temporal_thd):
            for lb, st, ed, score in v:
                s += str(int(lb)) + ',' + str(int(st)) + ',' + str(int(ed)) + ',' + str(score) + '\n'
            ff.write(s)
        
        
class ProgressMeter(object):

    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries),flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    for milestone in args.schedule:
        lr *= 0.1 if epoch >= milestone else 1.

    print(optimizer.param_groups)
    for param_group in optimizer.param_groups:
        print(type(param_group),  param_group['lr'])


def accuracy(output, target, topk=(1,), ignore=-1):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        if ignore != -1:
            if ignore == 0:
                mask = target != 0
            elif ignore == 1:
                mask = target == 0
            output = output[mask, :]
            target = target[mask]
            batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def get_proposal(brr): # return [L, R]
    arr = np.append(brr, 0)  #Add 0 at the end to handle consecutive 1s at the end
    proposal, start = [], None
    for i in range(arr.shape[0]):
        if arr[i] == 1:
            if start is None:
                start = i
        elif start is not None:
            proposal.append([start, i])
            start = None
    return proposal

def temporal_nms(actions, iou_threshold):
    """
    Implementation of Temporal Non-Maximum Suppression (NMS) for action detection in videos.
    
    Parameters:
    - actions: List of actions, each action formatted as (label, start, end, confidence)
    - iou_threshold: IOU threshold for suppression.
    
    Returns:
    - List of indices of actions that are retained.
    """
    if len(actions) == 0:
        return []
    
    # Convert to numpy array for easier manipulation
    actions = np.array(actions, dtype=np.float32)
    
    starts = actions[:, 1].astype(np.int32)
    ends = actions[:, 2].astype(np.int32)
    scores = actions[:, 3].astype(float)
    
    # Calculate the duration and area for each action
    durations = ends - starts
    area = durations
    
    # Sort indices by scores in descending order
    indices = np.argsort(scores)[::-1]
    
    keep = []
    
    while indices.size > 0:
        i = indices[0]
        keep.append(i)
        
        # Calculate the intersection times between the current action and all other actions
        tt1 = np.maximum(starts[i], starts[indices[1:]])
        tt2 = np.minimum(ends[i], ends[indices[1:]])
        
        # Calculate the duration of the intersections
        intersection = np.maximum(0.0, tt2 - tt1)
        
        # Keep actions where IOU is below the threshold
        iou = intersection / (area[i] + area[indices[1:]] - intersection)
        
        remaining = np.where(iou <= iou_threshold)[0]
        indices = indices[remaining + 1]
    return actions[keep].tolist()


if __name__ == '__main__':
    seed = 0
    random.seed(seed)         # Python随机库的种子
    np.random.seed(seed)      # NumPy随机库的种子
    torch.manual_seed(seed)   # PyTorch随机库的种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    main()

