import argparse, time
import os, random, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from dataset import get_pretraining_set
from model.loss import *
from tools import AverageMeter, sum_para_cnt

global ws
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0005, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[351], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--checkpoint-path', default='./checkpoint', type=str)

parser.add_argument('--pre-dataset', default='ntu60', type=str,
                    help='which dataset to use for self supervised training (ntu60 or ntu120)')
parser.add_argument('--protocol', default='cross_subject', type=str,
                    help='training protocol cross_view/cross_subject/cross_setup')
parser.add_argument('--moda', default='joint', type=str,
                    help='joint, motion , bone')
parser.add_argument('--backbone', default='DSTE', type=str,
                    help='DSTE or STTR')
                    
parser.add_argument('--padding', default='confidence', type=str,
                    help='set padding method to NTU2D, confidence/zero is valid.')

parser.add_argument('--ver', default='confidence', type=str,
                    help='set padding method to NTU2D, confidence/zero is valid.')



def main():
    args = parser.parse_args()
    # pretraining dataset and protocol
    from options import options_pretraining as options 
    if args.pre_dataset == 'pku_v2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_v2_xsub()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.pre_dataset == 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.pre_dataset == 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()
    elif args.pre_dataset == 'ntu60_2d' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject_2d(padding=args.padding)
    elif args.pre_dataset == 'ntu60_2d' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view_2d()
    elif args.pre_dataset == 'ntu120_2d' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject_2d(padding=args.padding)
    elif args.pre_dataset == 'ntu120_2d' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup_2d(padding=args.padding)
    elif args.pre_dataset == 'uav' and args.protocol == 'cross_subject':
        opts = options.opts_uav_cross_subject(ver=args.ver)
        
        
    # create model
    
    if args.backbone == 'DSTE':
        from model.DSTE import USDRL
        model = USDRL(**opts.encoder_args, modality=args.moda)
    elif args.backbone == 'STTR':
        from model.STTR import USDRL
        model = USDRL(**opts.encoder_args, modality=args.moda)
    else:
        print('backbone must be DSTE or STTR')
        exit(0) 

    print("options",opts.train_feeder_args)
    print("options",opts.encoder_args)
    print(model)
    print(args)
    print('para count:', sum_para_cnt(model)/1e6, 'M')

    #exit(0)
    model = torch.nn.DataParallel(model)
    model = model.cuda()      

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if torch.cuda.is_available():
                checkpoint = torch.load(args.resume)
            else:
                print('CUDA Error: torch.cuda.is_available() == False')
                exit(0)
                #checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
            del checkpoint
            torch.cuda.empty_cache()
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    ## Data loading code
    train_dataset = get_pretraining_set(opts)
    trainloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 8,
            'pin_memory': True,
            'prefetch_factor': 4,
            'persistent_workers': True
    }
    train_loader = torch.utils.data.DataLoader(train_dataset,  **trainloader_params)
    
    
    writer = SummaryWriter(args.checkpoint_path)

    scaler = torch.cuda.amp.GradScaler()
    print(ws)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        # train for one epoch
        
        st_epoch = time.time()
        loss = train(scaler, train_loader, model, criterion, optimizer, epoch, args)
        print('epoch ' +str(epoch) + ' time:', time.time()-st_epoch, '\n')
        
        writer.add_scalar('train_loss', loss.avg, global_step=epoch)
        if epoch % 50 == 0 and epoch > 1:
            save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=args.checkpoint_path+'/checkpoint_{:04d}.pth.tar'.format(epoch,loss.avg))


def train(scaler, train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses,],
        prefix="Epoch: [{}] Lr_rate [{}]".format(epoch,optimizer.param_groups[0]['lr']))

    # switch to train mode
    model.train()
    end = time.time()
    for i, (data_v1, data_v2, data_v3, data_v4) in enumerate(train_loader):
        # measure data loading time
        for k in loss_rcd.keys():
            loss_rcd[k].reset()
        data_time.update(time.time() - end)
        if torch.cuda.is_available():
            data_v1 = data_v1.float().cuda()
            data_v2 = data_v2.float().cuda()
            data_v3 = data_v3.float().cuda()
            data_v4 = data_v4.float().cuda()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            
            z_t_list, z_s_list, z_i_list = model(data_v1, data_v2, data_v3, data_v4) # [512, 4096]

            # Similarity
            sim_t = similarity(z_t_list, criterion)
            sim_s = similarity(z_s_list, criterion)
            sim_i = similarity(z_i_list, criterion)
            sim = sim_i + 0.5 * (sim_t + sim_s)

            B, _ = z_t_list[0].shape

            # Variance & AutoCov
            vac_t = sum([v_ac(x) for x in z_t_list])
            vac_s = sum([v_ac(x) for x in z_s_list])
            vac_i = sum([v_ac(x) for x in z_i_list])
            vac = vac_i + 0.5 * (vac_t + vac_s)

            # cross correlation = Invariance + Reudce Redundancy
            # xcorr_t = cal_xc(z_t_list)
            # xcorr_s = cal_xc(z_s_list)
            # xcorr_i = cal_xc(z_i_list)
            xcorr = torch.tensor(0).cuda() #xcorr_i + 0.5 * (xcorr_s + xcorr_t) 

            # Total loss, Multi-Grained Feature Decorrelation
            loss = sim * ws['sim'] + vac * ws['vac'] # + xcorr * ws['xcorr']
            if loss > 1e4 or torch.isnan(loss).any() or torch.isinf(loss).any():
                print(f'{loss} Model collapse, Exit 0')
                exit(0)
        losses.update(loss.item(), B)
        loss_rcd['vac'].update(vac.item(), B)        
        loss_rcd['sim'].update(sim.item(), B)
        loss_rcd['xcorr'].update(xcorr.item(), B)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()
        if i + 1 == len(train_loader):
            progress.display(i)
            s = str(epoch) + '\t'
            for k in loss_rcd.keys():
                s += loss_rcd[k].get_str() + '  '
            print(s)
    return losses


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

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



if __name__ == '__main__':
    #exit(0)
    #seed = 0
    #random.seed(seed)         # Python随机库的种子
    #np.random.seed(seed)      # NumPy随机库的种子
    #torch.manual_seed(seed)   # PyTorch随机库的种子
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed(seed)
    #    torch.cuda.manual_seed_all(seed)  # 如果使用多GPU
    #    torch.backends.cudnn.deterministic = True
    #    torch.backends.cudnn.benchmark = False
    ws = {
        'sim': 5,
        'vac': 1.,
        'xcorr': 0, # 1e-3
        }
    args = parser.parse_args()
    loss_rcd= {}
    for k in ws.keys():
        loss_rcd[k] = AverageMeter(k, '.5e')
    main()
    # 