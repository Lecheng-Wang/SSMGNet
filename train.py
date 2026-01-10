# encoding = utf-8

# @Author     ：Lecheng Wang
# @Time       : 2026/1/10 1:05 (Revised)
# @Function   : main control pannel
# @Description: train file


import os
import csv
import argparse
import torch
import random
import torch.nn             as nn
import numpy                as np
import torch.optim          as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional  as F
import torch.utils.data
#import tifffile

from model.model_spatial_sfcnm   import SegModel
from torch.utils.data            import DataLoader
from utils.dataset               import Labeled_Model_Dataset
from utils.metrics               import Evaluator
from utils.weight_init           import weights_init
from utils.focal                 import FocalLoss
from utils.sf_cnm_threshold      import Get_Mis_Threshold,Compute_Attention_Label
from tqdm                        import tqdm

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For muti-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
set_seed(42)  # Set seed at the beginning

parser = argparse.ArgumentParser(description="SFCNM")
parser.add_argument('--DATASET_PATH',        type=str,   default='./datasets/') # 检查是否修改
parser.add_argument('--MODE',                type=str,   default='single', choices=['single','muti','all'])
parser.add_argument('--GPU_ID',              type=int,   default=0)
parser.add_argument('--GPU_LIST',            type=str,   default='1,2')
parser.add_argument('--USE_CPU',             type=bool,  default=False)
parser.add_argument('--BANDS',               type=int,   default=10)
parser.add_argument('--NUM_CLASS',           type=int,   default=2+1)
parser.add_argument('--LR_STEP',             type=int,   default=1)
parser.add_argument('--STEP_RATIO',          type=float, default=0.94)
parser.add_argument('--INIT_LR',             type=float, default=5e-3)
parser.add_argument('--MOMENTUM',            type=float, default=0.9)
parser.add_argument('--WEIGHT_DECAY',        type=float, default=1e-4)
parser.add_argument('--BATCH_SIZE',          type=int,   default=8)
parser.add_argument('--START_EPOCH',         type=int,   default=1)
parser.add_argument('--EPOCHS',              type=int,   default=8)
parser.add_argument('--SEGMODEL_PRETRAIN',   type=str,   default='SegModel_pretrained.pth')
parser.add_argument('--MODEL_SAVE_EPOCHS',   type=int,   default=1)
parser.add_argument('--LOSS_TYPE',           type=str,   default='ce',     choices=['ce',  'focal'])
parser.add_argument('--OPTIMIZER_TYPE',      type=str,   default='adam',   choices=['adam','sgd'])
parser.add_argument('--LR_SCHEDULER',        type=str,   default='step',   choices=['poly','step', 'cos','exp'])
parser.add_argument('--INIT_TYPE',           type=str,   default='kaiming',choices=['kaiming','normal','xavier','orthogonal'])
args = parser.parse_args()


# Parallel Mode->
#        single: Training on only-one GPU
#        muti:  Parallel training on multiple GPUs in GPU_LIST
#        all:    Parallel training on all GPUs on this platform
if args.USE_CPU:
    device    = torch.device('cpu')
    print("当前代码已指定使用cpu进行训练\n", flush=True)

elif args.MODE=='single':
    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"即将在{device}上使用单卡训练模式\n", flush=True)
elif args.MODE=='muti':
    args.GPU_LIST = [int(gpu_id) for gpu_id in args.GPU_LIST.split(',')]
    if torch.cuda.is_available() and all(gpu < torch.cuda.device_count() for gpu in args.GPU_LIST):
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.GPU_LIST))
        device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
        print(f"即将在{args.GPU_LIST}号设备使用多卡训练模式\n", flush=True)
    else:
        device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"当前平台不满足多GPU模式，即将默认使用{device}进行训练\n", flush=True)
elif args.MODE=='all':
    device    = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()
    print(f"将使用当前平台全部{gpu_count}个GPU设备进行训练\n", flush=True)



def main (): 
    os.makedirs('pth_files', exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"GPU Memory Cleared: Allocated {torch.cuda.memory_allocated(device)/1024**2:.2f} MB")

    # 光谱样本矩阵
#    ill_cond_A = np.array([[0.2288, 0.0173, 0.0374, 0.0918, 0.0512, 0.1116, 0.0625, 0.0542, 0.0452, 0.0255],
#                           [0.2533, 0.0389, 0.0552, 0.1208, 0.0826, 0.1413, 0.0658, 0.0872, 0.0558, 0.0402],
#                           [0.3074, 0.0804, 0.0992, 0.1776, 0.1370, 0.1959, 0.0759, 0.1497, 0.0885, 0.0694],
#                           [0.3322, 0.0903, 0.1149, 0.1949, 0.1408, 0.2229, 0.0813, 0.1687, 0.1386, 0.0839],
#                           [0.2882, 0.0911, 0.1258, 0.2055, 0.1469, 0.2522, 0.1480, 0.1945, 0.1711, 0.0906],
#                           [0.1100, 0.0842, 0.1714, 0.2390, 0.1749, 0.3301, 0.2171, 0.2622, 0.1561, 0.0965],
#                           [0.0680, 0.0682, 0.1601, 0.2231, 0.1686, 0.2640, 0.1894, 0.2219, 0.1401, 0.0832],
#                           [0.5146, 0.6535, 0.5482, 0.5119, 0.4310, 0.4616, 0.3830, 0.4905, 0.6301, 0.6093],
#                           [0.0706, 0.0655, 0.0787, 0.0533, 0.1147, 0.0807, 0.0354, 0.0317, 0.0280, 0.1600],
#                           [0.7364, 0.4886, 0.3666, 0.4264, 0.4392, 0.3725, 0.2589, 0.3635, 0.3618, 0.4186]])

    ill_cond_A = np.array([
                [0.025108, 0.044775, 0.022713, 0.054517, 0.024255, 0.263413, 0.897558, 0.083041, 0.051523, 0.447467],
                [0.039021, 0.065726, 0.041806, 0.077207, 0.041771, 0.281572, 0.906344, 0.111226, 0.092698, 0.477236],
                [0.069077, 0.102917, 0.077322, 0.119946, 0.078582, 0.304236, 0.903852, 0.165515, 0.191769, 0.511057],
                [0.082601, 0.113206, 0.087972, 0.137899, 0.092967, 0.302216, 0.902451, 0.18202 , 0.192265, 0.536082],
                [0.111036, 0.139278, 0.096018, 0.146704, 0.098742, 0.231165, 0.737299, 0.199735, 0.150488, 0.497987],
                [0.110326, 0.150545, 0.140082, 0.20488 , 0.116332, 0.008532, 0.030921, 0.263537, 0.018009, 0.012086],
                [0.090256, 0.125565, 0.124102, 0.174439, 0.098414, 0.00814 , 0.032178, 0.232524, 0.019361, 0.01256],
                [0.608871, 0.597479, 0.526987, 0.49861 , 0.533335, 0.625007, 0.691949, 0.537595, 0.60378 , 0.654422],
                [0.119021, 0.078705, 0.108357, 0.080463, 0.103712, 0.094274, 0.06156 , 0.273285, 0.064751, 0.119046],
                [0.390516, 0.407727, 0.355867, 0.367505, 0.403623, 0.972147, 0.96713 , 0.385322, 0.894642, 0.976719]])    
    # 计算误分类物质
    mis_threshold = Get_Mis_Threshold(ill_cond_A)
    mis_threshold = mis_threshold.to(device)
    
    # 构建语义分割网络
    model = SegModel(bands=args.BANDS, num_classes=args.NUM_CLASS)
    if args.MODE in ['muti','all'] and (not args.USE_CPU):
        model = nn.DataParallel(model)
    model = model.to(device)

    if args.SEGMODEL_PRETRAIN:
        if os.path.isfile(args.SEGMODEL_PRETRAIN):
            print(f"=> Loading pretrained model from {args.SEGMODEL_PRETRAIN}")
            checkpoint = torch.load(args.SEGMODEL_PRETRAIN, map_location='cpu')            
            pretrained_dict = checkpoint
            if 'state_dict' in checkpoint:
                pretrained_dict = checkpoint['state_dict']
            pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
            model.load_state_dict(pretrained_dict, strict=False)
            print("=> Loaded pretrained weights (strict=False)")
    else:
        print(f"=> No pretrained model found at '{args.SEGMODEL_PRETRAIN}'")
        weights_init(model, init_type=args.INIT_TYPE)


    # 类别平衡权重
    weight    = np.array([0.03247, 0.25926, 0.70827], np.float32)
    weight    = torch.from_numpy(weight.astype(np.float32)).to(device)

    # 注意力优化损失函数选择
    atten_criterion = nn.MSELoss()

    # 分割网络优化损失函数选择
    if args.LOSS_TYPE == 'ce':
        seg_criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=-1, reduction='mean')
    elif args.LOSS_TYPE == 'focal':
        seg_criterion = FocalLoss(alpha=weight, gamma=2.0, ignore_index=-1, reduction='mean')
    else:
        raise NotImplementedError('loss type [%s] is not implemented,ce/focal is supported!' %args.LOSS_TYPE)
    


    # 优化器选择
    if args.OPTIMIZER_TYPE == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.INIT_LR, momentum=args.MOMENTUM, weight_decay=args.WEIGHT_DECAY)
    elif args.OPTIMIZER_TYPE == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.INIT_LR, weight_decay=args.WEIGHT_DECAY)
    else:
        raise NotImplementedError('optimizer type [%s] is not implemented,sgd and adam is supported!' %args.OPTIMIZER_TYPE)

    # 学习率衰减方式选择
    if args.LR_SCHEDULER == 'step':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.LR_STEP, gamma=args.STEP_RATIO)
    elif args.LR_SCHEDULER == 'exp':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.STEP_RATIO)
    elif args.LR_SCHEDULER == 'cos':
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.EPOCHS)
    elif args.LR_SCHEDULER == 'poly':
        lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=args.EPOCHS, power=2.5)
    else:
        raise NotImplementedError('lr scheduler type [%s] is not implemented,cos,step,poly,exp is supported!' %args.LR_SCHEDULER)

    # 读取训练集和测试集目录
    with open(os.path.join(args.DATASET_PATH, "annotations/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.DATASET_PATH, "annotations/val.txt"),"r") as f:
        test_lines  = f.readlines()

    # 将训练集和测试集目录转化为Datasets，再转化为batchsize个dataloader
    train_datasets  = Labeled_Model_Dataset(train_lines, args.DATASET_PATH)
    test_datasets   = Labeled_Model_Dataset(test_lines,  args.DATASET_PATH)
    train_loader    = DataLoader(train_datasets,shuffle=True, batch_size=args.BATCH_SIZE,num_workers=0,pin_memory=True,drop_last=True)
    test_loader     = DataLoader(test_datasets, shuffle=False,batch_size=args.BATCH_SIZE,num_workers=0,pin_memory=True,drop_last=False)



    cudnn.benchmark = True

    # 创建训练过程的对象
    trainer = Trainer(args, model, ill_cond_A, seg_criterion, atten_criterion, optimizer, train_loader, test_loader, mis_threshold)

    print('Starting Epoch:', trainer.args.START_EPOCH)
    print('Total Epoches:',  trainer.args.EPOCHS)

    with open(f'metric_training_log.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch','train_loss','train_atten','val_loss','val_atten','Acc','Kappa','mIoU','mIoU0','mIoU1','mIoU2','FWIoU','Precision','Precision0','Precision1','Precision2','Recall','Recall0','Recall1','Recall2','F1_score','F1_score0','F1_score1','F1_score2','F2_score','F2_score0','F2_score1','F2_score2'])

    # 训练EPOCHS个周期
    print(f"Start training on device:{device}...")
    init_weight_value = 6
    for epoch in range(trainer.args.EPOCHS):
        if (epoch + 1) % 10 == 0:
            init_weight_value /= 2
        train_loss,train_atten = trainer.training(epoch,init_weight_value)
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]
        print("Current learning rate is:", current_lr)
        print("Training over.\n")

        print(f"Start validating on device:{device}......")
        val_loss,val_atten,Acc,Kappa,mIoU,mIoU0,mIoU1,mIoU2,FWIoU,Precision,Precision0,Precision1,Precision2,Recall,Recall0,Recall1,Recall2,F1_score,F1_score0,F1_score1,F1_score2,F2_score,F2_score0,F2_score1,F2_score2=trainer.validation(epoch)
        print("Validating over.\n")

        if (epoch + 1) % args.MODEL_SAVE_EPOCHS == 0:
            torch.save(model.state_dict(),'pth_files/epoch%d-loss%.3f-val_loss%.3f.pth'%(epoch+1,train_loss,val_loss))

        with open(f'metric_training_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch+1,train_loss,train_atten,val_loss,val_atten,Acc,Kappa,mIoU,mIoU0,mIoU1,mIoU2,FWIoU,Precision,Precision0,Precision1,Precision2,Recall,Recall0,Recall1,Recall2,F1_score,F1_score0,F1_score1,F1_score2,F2_score,F2_score0,F2_score1,F2_score2])



class Trainer(object):
    def __init__(self, args, model, Matrix, seg_criterion, atten_criterion, optimizer, train_loader, val_loader, threshold):
        self.args         = args
        self.model        = model
        self.Matrix       = Matrix
        self.seg_crit     = seg_criterion
        self.att_crit     = atten_criterion
        self.optimizer    = optimizer
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.evaluator    = Evaluator(self.args.NUM_CLASS)
        self.threshold    = threshold

    def training(self, epoch, weight_value=6):
        self.model.train()
        train_loss      = 0.0
        atten_loss      = 0.0
        train_loader    = tqdm(self.train_loader)
        num_batch       = len(self.train_loader)

        for i, data in enumerate(train_loader):
            img, lbl    = data
            img         = img.float().to(device)
            lbl         = lbl.long().to(device)

            att_lbl1,_  = Compute_Attention_Label(img, self.Matrix, self.threshold)
            att_lbl1    = att_lbl1.to(img.device)

            output,_,w4 = self.model(img)
            output      = output.float()

            loss1       = self.seg_crit(output, lbl)
            loss5       = self.att_crit(w4, att_lbl1)

            loss        = loss1 + weight_value*loss5

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_loss  += loss1.item()
            atten_loss  += loss5.item()
            train_loader.set_description('Train loss: %.3f' % (train_loss / (i + 1)))
        print('Epoch: %d, numImages: %5d' % (epoch+1, num_batch * self.args.BATCH_SIZE))
        print('Train Loss: %.3f' % (train_loss / num_batch))
        return train_loss/num_batch, atten_loss/num_batch

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        val_loss    = 0.0
        atten_loss  = 0.0
        val_loader  = tqdm(self.val_loader)
        num_batch   = len(self.val_loader)
        
        with torch.no_grad():
            for i, sample in enumerate(val_loader):
                image, label = sample
                image        = image.float().to(device)
                label        = label.long().to(device)

                att_lbl1,_   = Compute_Attention_Label(image, self.Matrix, self.threshold)
                att_lbl1     = att_lbl1.to(image.device)
                output,_,w4  = self.model(image)
                output       = output.float()

                loss1        = self.seg_crit(output, label)
                loss5        = self.att_crit(w4, att_lbl1)
                val_loss    +=  loss1.item()
                atten_loss  += loss5.item()
                val_loader.set_description('Val loss: %.3f' % (val_loss / (i + 1)))
                pred         = output.data.cpu().numpy()
                label        = label.cpu().numpy()
                pred         = np.argmax(pred, axis=1)
                self.evaluator.add_batch(label, pred)

        Acc                              = self.evaluator.OverAll_Accuracy()
        Kappa                            = self.evaluator.Kappa()

        mIoU,    IoU                     = self.evaluator.mean_Intersection_over_Union()
        mIoU0,     mIoU1,     mIoU2      = IoU

        FWIoU                            = self.evaluator.Frequency_Weighted_Intersection_over_Union()

        mPrecision,Precision             = self.evaluator.Precision()
        Precision0,Precision1,Precision2 = Precision

        mRecall,   Recall                = self.evaluator.Recall()
        Recall0,   Recall1,   Recall2    = Recall

        mF1_score,F1_score               = self.evaluator.F1_Score()
        F1_score0, F1_score1, F1_score2  = F1_score

        mF2_score,F2_score               = self.evaluator.F2_Score()
        F2_score0, F2_score1, F2_score2  = F2_score

        print('Validation Result:')
        print('Epoch:%d, numImages: %5d' % (epoch+1, num_batch * self.args.BATCH_SIZE))
        print("Epoch:{}, Acc:{:.4f},Kappa:{:.4f}, mIoU:{:.4f}, FWIoU: {:.4f},\nPrecision: {:.4f}, Recall: {:.4f}, f1_score: {:.4f}, f2_score: {:.4f}.".format(epoch+1,Acc,Kappa,mIoU,FWIoU,mPrecision,mRecall,mF1_score,mF2_score))
        print('Val Loss: %.3f' % (val_loss / num_batch)) 

        return val_loss/num_batch,atten_loss/num_batch,Acc,Kappa,mIoU,mIoU0,mIoU1,mIoU2,FWIoU,mPrecision,Precision0,Precision1,Precision2,mRecall,Recall0,Recall1,Recall2,mF1_score,F1_score0,F1_score1,F1_score2,mF2_score,F2_score0,F2_score1,F2_score2

if __name__ == '__main__':

    main()

