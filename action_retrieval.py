from dataset import get_finetune_training_set, get_finetune_validation_set
import argparse, time, os, random
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.utils.data
import numpy as np
from sklearnex import patch_sklearn, unpatch_sklearn
patch_sklearn()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score
from tools import sum_para_cnt, remove_prefix


# change for action recogniton

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=80, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=30., type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--schedule', default=[50, 70, ], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by a ratio)')

parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')

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
parser.add_argument('--knn-neighbours', default=1, type=int,
                    help='number of neighbours used for KNN.')

best_acc1 = 0
def load_pretrained(model, pretrained):
    if os.path.isfile(pretrained):
        print("=> loading checkpoint '{}'".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")

        # rename pre-trained keys
        state_dict = checkpoint['state_dict']
        
        state_dict = remove_prefix(state_dict)
        

        
        msg = model.load_state_dict(state_dict, strict=False)
        print("message", msg)
        
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

        print("=> loaded pre-trained model '{}'".format(pretrained))
    else:
        print("=> no checkpoint found at '{}'".format(pretrained))


def knn(data_train, data_test, label_train, label_test, nn=9):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)
    print("Number of KNN Neighbours = ", nn)
    print("training feature and labels", data_train.shape, len(label_train))
    print("test feature and labels", data_test.shape, len(label_test))
    # preprocessing.normalize(data_train)
    # packge <prereocessing> use cpu, we can use gpu(CUDA) to accelaerate this operation.
    # preprocessing.normalize(data_test)
    Xtr_Norm = data_train
    Xte_Norm = data_test
    knn = KNeighborsClassifier(n_neighbors=nn,
                               metric='cosine',
                               n_jobs=24)  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})

    a = time.time()
    knn.fit(Xtr_Norm, label_train)
    b = time.time()
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)

    return acc, str(b-a)[:5]


def test_extract_hidden(model, data_train, data_eval, args):
    
    model.eval()
    print("Extracting training features")
    label_train_list = []
    hidden_array_train_list = []
    for ith, (jt, js, bt, bs, mt, ms, label) in tqdm(enumerate(data_train)):
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)  
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True) 
        label = label.long().cuda()


        en_hi = model(jt, js, bt, bs, mt, ms, knn_eval=True)
         
        label_train_list.append(label)
        hidden_array_train_list.append(en_hi)
    
    label_train = torch.cat(label_train_list).cpu().numpy()
    hidden_array_train = torch.nn.functional.normalize(torch.cat(hidden_array_train_list), p=2, dim=1).cpu().numpy()
    print("Extracting validation features")
    label_eval_list = []
    hidden_array_eval_list = []

    for ith, (jt, js, bt, bs, mt, ms, label) in tqdm(enumerate(data_eval)):
        jt = jt.float().cuda(non_blocking=True)
        js = js.float().cuda(non_blocking=True)
        bt = bt.float().cuda(non_blocking=True)
        bs = bs.float().cuda(non_blocking=True)  
        mt = mt.float().cuda(non_blocking=True)
        ms = ms.float().cuda(non_blocking=True)  
        label = label.long().cuda()
        en_hi = model(jt, js, bt, bs, mt, ms, knn_eval=True)

        label_eval_list.append(label)
        hidden_array_eval_list.append(en_hi)
    label_eval = torch.cat(label_eval_list).cpu().numpy()
    hidden_array_eval = torch.nn.functional.normalize(torch.cat(hidden_array_eval_list), p=2, dim=1).cpu().numpy()

    return hidden_array_train, hidden_array_eval, label_train, label_eval




def clustering_knn_acc(model, train_loader, eval_loader, knn_neighbours=1, args=None):
    with torch.no_grad():
        hi_train, hi_eval, label_train, label_eval = test_extract_hidden(model, train_loader, eval_loader, args)
    knn_acc_1, time_cost = knn(hi_train, hi_eval, label_train, label_eval, nn=knn_neighbours)
    knn_acc_au = knn_acc_1

    return knn_acc_1, knn_acc_au, time_cost


def main():
    
    args = parser.parse_args()
    if not os.path.exists(args.pretrained):
        print(args.pretrained, ' not found!')
        exit(0)
    # Simply call main_worker function
    main_worker(args)


def main_worker(args):

    # training dataset
    from options  import options_downstream as options 
    if args.finetune_dataset== 'pku_v2' and args.protocol == 'cross_subject':
        opts = options.opts_pku_v2_xsub()
    elif args.finetune_dataset== 'ntu60' and args.protocol == 'cross_view':
        opts = options.opts_ntu_60_cross_view()
    elif args.finetune_dataset== 'ntu60' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_60_cross_subject()
    elif args.finetune_dataset== 'ntu120' and args.protocol == 'cross_setup':
        opts = options.opts_ntu_120_cross_setup()
    elif args.finetune_dataset== 'ntu120' and args.protocol == 'cross_subject':
        opts = options.opts_ntu_120_cross_subject()

  
    # create model
    print("=> creating model")
    
    if args.backbone == 'DSTE':
        from model.DSTE import Downstream
        model = Downstream(**opts.encoder_args)
    elif args.backbone == 'STTR':
        from model.STTR import Downstream
        model = Downstream(**opts.encoder_args)
    print(sum_para_cnt(model)/1e6)
    #print(model)
    print("options: \n",opts.encoder_args,'\n',opts.train_feeder_args,'\n',opts.test_feeder_args)
    

    # freeze all layers
    for _, param in model.named_parameters():
        param.requires_grad = False

    # load from pre-trained  model
    load_pretrained(model, args.pretrained)

    model = model.cuda()

    # Data loading code

    train_dataset = get_finetune_training_set(opts)
    val_dataset = get_finetune_validation_set(opts)
    trainloader_params = {
            'batch_size': args.batch_size,
            'shuffle': True,
            'num_workers': 1,
            'pin_memory': True,
            'prefetch_factor': 1,
            'persistent_workers': False
    }
    valloader_params = {
            'batch_size': args.batch_size,
            'shuffle': False,
            'num_workers': 1,
            'pin_memory': True,
            'prefetch_factor': 1,
            'persistent_workers': False
    }
    train_loader = torch.utils.data.DataLoader(train_dataset,  **trainloader_params)
    val_loader = torch.utils.data.DataLoader(val_dataset,  **valloader_params)

    # Extract frozen features of  the  pre-trained query encoder
    # train and evaluate a KNN  classifier on extracted features
    acc1, _, tc = clustering_knn_acc(model, train_loader, val_loader,
                                      knn_neighbours=args.knn_neighbours, args=args)
    print(args.pretrained, 'Knn time cost:' + tc + "s\tKnn Without AE= ", acc1)#, " Knn With AE=", acc_au)


if __name__ == '__main__':
    
    main()