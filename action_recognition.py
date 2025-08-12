import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import numpy as np

from tools import AverageMeter, remove_prefix, sum_para_cnt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
# change for action recogniton
from dataset import get_finetune_training_set,get_finetune_validation_set

global best_acc1
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

parser.add_argument('--finetune-dataset', default='ntu60', type=str,
                    help='which dataset to use for finetuning')

parser.add_argument('--protocol', default='cross_view', type=str,
                    help='traiining protocol of ntu')

parser.add_argument('--moda', default='joint', type=str,
                    help='joint, motion , bone')

parser.add_argument('--backbone', default='DSTE', type=str,
                    help='DSTE or STTR')

parser.add_argument('--padding', default='DSTE', type=str,
                    help='DSTE or STTR')


parser.add_argument('--ver', default='DSTE', type=str,
                    help='DSTE or STTR')


best_acc1 = 0



def load_encoder(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")
        # rename pre-trained keys
        state_dict = checkpoint['state_dict']
        state_dict = remove_prefix(state_dict)
        msg = model.load_state_dict(state_dict, strict=False)
        print("message",msg)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def load_pretrained(args, model):
    
    load_encoder(model,args.pretrained)
    finetune_encoder = True

    return finetune_encoder



def main():
    args = parser.parse_args()
    if not os.path.exists(args.pretrained):
        print(args.pretrained, ' not found!')
        exit(0)
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    global best_acc1

    # create model

    # training dataset
    from options  import options_downstream as options 
    if args.finetune_dataset == 'pku_v2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_v2_xsub()
    elif args.finetune_dataset== 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.finetune_dataset== 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.finetune_dataset== 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.finetune_dataset== 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.finetune_dataset== 'uav' and args.protocol == 'cross_subject':
        opts = options.opts_uav_cross_subject(ver=args.ver)
    elif args.finetune_dataset== 'ntu60_2d' and args.protocol == 'cross_subject':
        opts = options.opts_ntu2d_60_cross_subject(padding=args.padding)
    elif args.finetune_dataset== 'ntu120_2d' and args.protocol == 'cross_subject':
        opts = options.opts_ntu2d_120_cross_subject(padding=args.padding)
    elif args.finetune_dataset== 'ntu60_2d' and args.protocol == 'cross_view':
        opts = options.opts_ntu2d_60_cross_view(padding=args.padding)
    elif args.finetune_dataset== 'ntu120_2d' and args.protocol == 'cross_setup':
        opts = options.opts_ntu2d_120_cross_setup(padding=args.padding)
    if args.backbone == 'DSTE':
        from model.DSTE import Downstream
        model = Downstream(**opts.encoder_args, modality=args.moda)
    elif args.backbone == 'STTR':
        from model.STTR import Downstream
        model = Downstream(**opts.encoder_args, modality=args.moda)
    else:
        print('backbone must be DSTE or STTR')
        exit(0) 

    print(sum_para_cnt(model)/1e6, 'M')
    print("options",opts.encoder_args,opts.train_feeder_args,opts.test_feeder_args, '\n',args)

    if args.pretrained:
        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            #break
            if not name.startswith('fc'):
                param.requires_grad = False
            else:
                print('params',name)
        # init the fc layer
        model.fc.weight.data.normal_(mean=0.0, std=0.01)
        model.fc.bias.data.zero_()

    # load from pre-trained model
    finetune_encoder= load_pretrained(args, model)
    model = nn.DataParallel(model)
    model = model.cuda()
    
    
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
 
    #if args.pretrained:
    #    assert len(parameters) == 2  # fc.weight, fc.bias
        
    optimizer = torch.optim.SGD(parameters, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    for parm in optimizer.param_groups:
        print ("optimize parameters lr ",parm['lr'])


    ## Data loading code

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

    print('lr =', args.lr)
    for epoch in range(0, 10 + args.epochs):

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        if (epoch + 1) % 5 == 0:
            acc1 = validate(val_loader, model, criterion, args)
        else:
            acc1 = 0
            
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        if is_best:
            print("found new best accuracy:= ",acc1)
            best_acc1 = max(acc1, best_acc1)
            #state_dict = {
            #            'epoch': epoch + 1,
            #            'acc': best_acc1, 
            #            'state_dict': model.state_dict(),
            #            #'optimizer' : optimizer.state_dict(),
            #        }
            
        # sanity check 
        #if epoch == 0:
        #    if finetune_encoder:
        #        sanity_check_encoder(model.state_dict(), args.pretrained)
    print(args.pretrained, "class head Final best accuracy",best_acc1)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    #model.eval()
    model.train()
    
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
        output = model(jt, js, bt, bs, mt, ms, args)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), jt.size(0))
        top1.update(acc1[0], jt.size(0))
        top5.update(acc5[0], jt.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i + 1 == len(train_loader):
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (jt, js, bt, bs, mt, ms, target) in enumerate(val_loader):            
            jt = jt.float().cuda(non_blocking=True)
            js = js.float().cuda(non_blocking=True)
            bt = bt.float().cuda(non_blocking=True)
            bs = bs.float().cuda(non_blocking=True)
            mt = mt.float().cuda(non_blocking=True)
            ms = ms.float().cuda(non_blocking=True)
            target = target.long().cuda(non_blocking=True)
            # compute output
            output = model(jt, js, bt, bs, mt, ms, args)
            loss = criterion(output, target)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), jt.size(0))
            top1.update(acc1[0], jt.size(0))
            top5.update(acc5[0], jt.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i+ 1 == len(val_loader):
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        

    return top1.avg


def sanity_check_encoder(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = remove_prefix(checkpoint['state_dict'])
    state_dict = remove_prefix(state_dict)
    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k or k.find('projector') != -1:
            continue

        # name in pretrained model
        k_pre = 'module.' + k
        k_pre = k
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


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
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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

