import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from time import time
from torch.utils.tensorboard import SummaryWriter
import os
import sys

from utils.dataset import HumanMobilityDataset
from utils.data_process import diag_zero, y_dim_reduce

from models.GCN_3D import GCN_3D
from models.GCN_Transformer import GCN_Transformer
from models.PureTransformer import PureTransformer
from models.GCN_OD_Transformer import GCN_OD_Transformer
from models.GCN_n_Transformer import GCN_n_Transformer
from models.GCformer_nopatch import GCformer_nopatch

# 训练一个轮次
def train(model, train_loader, optimizer, criterion, epoch, device):
    model.train()
    train_loss = 0
    # count = 0
    for batch_idx, (x, A, y_true, x_timeF, y_timeF) in enumerate(train_loader):
        # x:C*T_in*V(region数量,=128)，历史5 interval内的io flow
        # A:T*V*V
        # y_true:C*T_out*V，未来1 interval内的io flow
        x = Variable(x).to(device)
        A = Variable(A).to(device)
        y_true = Variable(y_true).to(device)
        x_timeF = Variable(x_timeF).to(device)
        y_timeF = Variable(y_timeF).to(device)
        optimizer.zero_grad()
        # 模型运行
        y_pred = model(x, A, x_timeF, y_timeF)
        # ioFlow/ODFlow
        if len(y_pred.shape) == 3:
            y_true = y_dim_reduce(y_true)
        # 回归问题——预测和GT的io flow偏差
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        if batch_idx % 5 == 0:
            print("Epoch: %d [%d/%d (%.0f%%)]\tLoss: %.6f" % (
                epoch, batch_idx * len(x), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        train_loss += loss.item() * len(x)
        # count += len(x)
    # 返回rMSE
    train_loss = train_loss / len(train_loader.dataset)
    print("Epoch: %d \tTrain MSE: %.6f" % (epoch, train_loss))
    return train_loss


# 验证一个轮次
def val(model, data_loader, criterion_mse, criterion_mae, epoch, mode, device, is_diag_zero):
    model.eval()
    val_loss = np.zeros(2)
    # count = 0
    for batch_idx, (x, A, y_true, x_timeF, y_timeF) in enumerate(data_loader):
        x = Variable(x).to(device)
        A = Variable(A).to(device)
        y_true = Variable(y_true).to(device)
        x_timeF = Variable(x_timeF).to(device)
        y_timeF = Variable(y_timeF).to(device)
        # 模型运行
        y_pred = model(x, A, x_timeF, y_timeF)
        if is_diag_zero:
            y_true = diag_zero(y_true)
            y_pred = diag_zero(y_pred)
        # ioFlow/ODFlow
        if len(y_pred.shape) == 3:
            y_true = y_dim_reduce(y_true)
        loss_mse = criterion_mse(y_pred, y_true)
        loss_mae = criterion_mae(y_pred, y_true)
        val_loss[0] += loss_mse.item() * len(x)
        val_loss[1] += loss_mae.item() * len(x)
        # count += len(x)
    val_loss = val_loss / len(data_loader.dataset)
    print("Epoch: %d \t%s MSE: %.6f\tMAE: %.6f" % (epoch, mode, val_loss[0], val_loss[1]))
    return val_loss[0], val_loss[1]


def main():
    parser = argparse.ArgumentParser(description='Human Mobility Pred')
    parser.add_argument('--model', type=str, default='GCformer', metavar='M',
                        help='3DGCN / GCformer / transformer/ GCN-OD / GCN-n / GCformer-nopatch')
    parser.add_argument('--fusion_method', type=str, default='concat', metavar='FM',
                        help='concat / add / mul / biLinear / sum-pooling')
    parser.add_argument('--batch', type=int, default=32, metavar='B',
                        help='batch size')
    parser.add_argument('--epoch', type=int, default=5, metavar='E',
                        help='number of iterations')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--seed', type=int, default=12345, metavar='S',
                        help='random seed')
    parser.add_argument('--dropout', type=float, default=0.2, metavar='DO')
    parser.add_argument('--T_in', type=int, default=12, metavar='TI',
                        help='number of input time intervals in a training sample')
    parser.add_argument('--T_out', type=int, default=2, metavar='TO',
                        help='number of output time intervals in a training sample')
    parser.add_argument('--T_label', type=int, default=1, metavar='TL',
                        help='number of labeled time intervals in out time intervals')
    parser.add_argument('--load', type=int, default='0', metavar='LD',
                        help='load checkpoint/epoch_x.pt (x>0)')
    parser.add_argument('--normalize', type=int, default=0,
                        help='way of normalize (0-no,1--MinMax,2--Log,3--Mul,4--Add)')
    parser.add_argument('--norm_param', type=float, default=1,
                        help='normalize 3,4 param')
    parser.add_argument('--data', type=str, default='GZZone',
                        help='location of the data file')
    parser.add_argument('--data_X', type=str, default='OD',
                        help='value: ioFlow(C=2)/OD(C=V)')
    parser.add_argument('--train_split', type=float, default=0.6)
    parser.add_argument('--patch_size', type=int, default=512, help='patch size')
    parser.add_argument('--patch_method', type=str, default='embed', metavar='PM',help='embed / divide')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--cuda', type=int, default=0, help='cuda')
    parser.add_argument('--model_store', action='store_true', help='save model', default=False)
    parser.add_argument('--writer_store', action='store_true', help='save model', default=False)
    parser.add_argument('--zero_diag', action='store_true', help='save model', default=False)
    args = parser.parse_args()

    # seed设置
    torch.manual_seed(args.seed)
    # cuda设置
    if torch.cuda.is_available():
        device = torch.device('cuda', args.cuda)
    else:
        device = torch.device("cpu")

    # 0. 路径参数设置
    # 保存文件路径
    param_str = ''
    if args.fusion_method != 'concat':
        param_str = param_str + '_F' + args.fusion_method
    if args.patch_method != 'embed':
        param_str = param_str + '_P' + args.patch_method
    filename = args.model + '_' + args.data + '_i' + str(args.T_in) + 'o' + str(args.T_out) \
               + '_l' + str(args.T_label) + '_ep' + str(args.epoch) + param_str
    # path to save the train log (SumWriter)
    save_log = os.path.join('checkpoint', 'logs', filename)
    # path to save the train log (Text)
    save_txt = os.path.join('checkpoint', 'logs', filename + '.txt')
    # path to save the final model
    save_model = os.path.join('checkpoint', 'saves', filename + '.pt')

    # 原始数据路径
    X_path = os.path.join('data', args.data, args.data_X + '_' + args.data + '.npy')
    A_path = os.path.join('data', args.data, 'OD_' + args.data + '.npy')
    timeF_path = os.path.join('data', args.data, 'timeF_' + args.data + '.npy')

    # 添加tensorboard
    if args.writer_store:
        writer = SummaryWriter(save_log)
    else:
        writer = None
    # 输出训练日志文件
    file_save_txt = open(save_txt, 'w')
    print("-----------Start of model train-----------")
    # 把默认的“板子” - 命令行做个备份，以便可以改回来
    __console__ = sys.stdout
    sys.stdout = file_save_txt

    # 1. 数据集加载
    train_data = HumanMobilityDataset(args.data, X_path, A_path, timeF_path,
                                      T_i=args.T_in, T_o=args.T_out,
                                      type='train', train_split=args.train_split,
                                      normalize=args.normalize,
                                      norm_param=args.norm_param)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)

    val_data = HumanMobilityDataset(args.data, X_path, A_path, timeF_path,
                                    T_i=args.T_in, T_o=args.T_out,
                                    type='val', train_split=args.train_split,
                                    normalize=args.normalize,
                                    norm_param=args.norm_param)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch, shuffle=False)

    test_data = HumanMobilityDataset(args.data, X_path, A_path, timeF_path,
                                     T_i=args.T_in, T_o=args.T_out,
                                     type='test', train_split=args.train_split,
                                     normalize=args.normalize,
                                     norm_param=args.norm_param)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch, shuffle=False)
    # 数据集参数
    C = train_data.numChannels()
    V = train_data.numRegions()

    # 2. 构建模型
    # '3DGCN / GCformer / transformer/ GCN-OD / GCN-n'
    model_dict = {
        '3DGCN': GCN_3D,
        'GCformer': GCN_Transformer,
        'transformer': PureTransformer,
        'GCN-OD': GCN_OD_Transformer,
        'GCN-n': GCN_n_Transformer,
        'GCformer-nopatch': GCformer_nopatch
    }
    model = None
    if args.model in model_dict:
        model = model_dict[args.model](C=C, V=V, device=device,
                                       patch_size=args.patch_size,
                                       d_model=args.d_model,
                                       d_ff=args.d_ff,
                                       T_i=args.T_in,
                                       T_o=args.T_out,
                                       T_label_i=args.T_label,
                                       data=args.data,
                                       fusion=args.fusion_method,
                                       patch=args.patch_method,
                                       dropout=args.dropout)
    model = model.to(device)

    # 继续之前的checkpoint训练
    if args.load > 0:
        load_filepath = os.path.join('checkpoint', 'epoch_' + str(args.load) + '.pt')
        model.load_state_dict(torch.load(load_filepath))

    # 3. 定义优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss(reduction='mean')
    criterion2 = nn.L1Loss(reduction='mean')
    val_loss_min = [100000, 100000]
    val_min_ep = [0, 0]
    total_val_loss = [0, 0]
    val_epoch = 0
    test_loss_min = [100000, 100000]
    test_min_ep = [0, 0]
    total_test_loss = [0, 0]
    test_epoch = 0
    total_time_used = 0

    print("-----------start training-----------")
    # 4. 开始训练，从load开始训练epoch个轮次
    for epoch in range(1 + args.load, args.epoch + 1 + args.load):
        # 计时器 start
        start = time()
        # 4.1 训练
        train_loss = train(model, train_loader, optimizer, criterion, epoch, device=device)
        if args.writer_store:
            writer.add_scalar('train loss', train_loss, epoch)
        # 4.2 验证
        val_mse, val_mae = val(model, val_loader, criterion, criterion2, epoch, mode='Val', device=device, is_diag_zero=args.zero_diag)
        if args.writer_store:
            writer.add_scalar('eval mse', val_mse, epoch)
            writer.add_scalar('eval mae', val_mae, epoch)
        # 计时器 stop
        stop = time()
        # 计算时间消耗
        print("Time used: %.3f\n" % (stop - start))
        total_time_used += (stop - start)

        # 计算平均val损失
        if epoch > 4:
            total_val_loss[0] += val_mse
            total_val_loss[1] += val_mae
            val_epoch += 1
        # 记录最小val损失
        if val_mse < val_loss_min[0]:
            if args.model_store:
                with open(save_model, 'wb') as f:
                    torch.save(model.state_dict(), f)
            val_loss_min[0] = val_mse
            val_min_ep[0] = epoch
        if val_mae < val_loss_min[1]:
            val_loss_min[1] = val_mae
            val_min_ep[1] = epoch

        # 4.3 测试(每五轮进行一次)
        if epoch % 5 == 0:
            test_mse, test_mae = val(model, test_loader, criterion, criterion2, epoch, mode='Test', device=device, is_diag_zero=args.zero_diag)
            if args.writer_store:
                writer.add_scalar('test mse', test_mse, epoch)
                writer.add_scalar('test mae', test_mae, epoch)
            # 计算平均test损失
            total_test_loss[0] += test_mse
            total_test_loss[1] += test_mae
            test_epoch += 1
            # 记录最小test损失
            if test_mse < test_loss_min[0]:
                test_loss_min[0] = test_mse
                test_min_ep[0] = epoch
            if test_mae < test_loss_min[1]:
                test_loss_min[1] = test_mae
                test_min_ep[1] = epoch

    # 训练结果——最小损失
    print("-----------End of model train (min loss values)-----------")
    print("Val MSE: %.6f \tTest MSE: %.6f" % (val_loss_min[0], test_loss_min[0]))
    print("epoch: %d \tepoch: %d" % (val_min_ep[0], test_min_ep[0]))
    print("Val MAE: %.6f \tTest MAE: %.6f" % (val_loss_min[1], test_loss_min[1]))
    print("epoch: %d \tepoch: %d\n" % (val_min_ep[1], test_min_ep[1]))
    # 训练结果——平均损失
    print("-----------End of model train (avg loss values)-----------")
    print("Val MSE: %.6f \tTest MSE: %.6f" % (total_val_loss[0]/val_epoch, total_test_loss[0]/test_epoch))
    print("Val MAE: %.6f \tTest MAE: %.6f\n" % (total_val_loss[1]/val_epoch, total_test_loss[1]/test_epoch))
    # 训练结果——平均时间消耗
    print("-----------End of model train (avg time used)-----------")
    print("Average time used (per epoch): %.5f\n" % (total_time_used / args.epoch))

    # 保存训练结果
    if args.model_store:
        save_filepath = os.path.join('checkpoint', 'epoch_' + str(epoch) + '.pt')
        torch.save(model.state_dict(), save_filepath)

    # 关闭tensorboard
    if args.writer_store:
        writer.close()
    file_save_txt.close()
    # 恢复stdout
    sys.stdout = __console__
    print("-----------End of model train-----------")


if __name__ == '__main__':
    main()
