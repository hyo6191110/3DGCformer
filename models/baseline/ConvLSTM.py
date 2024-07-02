import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
ConvLSTM单元
"""
class ConvLSTMBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.
        ----------
        Parameters
        ----------
        input_dim: Number of channels of input tensor.
        hidden_dim: Number of channels of hidden state.
        kernel_size: Size of the Conv kernel.
        bias: Whether or not to add the bias.
        """

        super(ConvLSTMBlock, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias
        # 保证在Conv过程中输入tensor的 h,w不变
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        # 定义ConvBlock
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,  # i、f、o、g门一起计算，然后split分开
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        """
        Forward Propagation.
        ----------
        Parameters
        ----------
        input_tensor:x_t
        cur_state:c_(t-1),h_(t-1)
        """
        # 每个timestamp包含两个状态张量：h和c
        h_cur, c_cur = cur_state
        # concatenate along channel axis 把输入张量x与状态张量h沿通道维度串联
        combined = torch.cat([input_tensor, h_cur], dim=1)
        # i、f、o、g门一起进行计算
        combined_conv = self.conv(combined)
        # 分开i、f、o、g门
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # 更新状态张量c_(t-1),h_(t-1)为c_t,h_t
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        # 输出当前时间t的两个状态张量c_t,h_t
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize c_0,h_0
        ----------
        Parameters
        ----------
        batch_size:bsz
        image_size:h,w
        """
        height, width = image_size
        init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        init_c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return (init_h, init_c)


"""
以ConvLSTM单元为基础的，简单堆叠模块的LSTM网络
----------
Input
----------
A tensor of size [B, T, C, H, W] or [T, B, C, H, W]
----------
Output
----------
return 2 lists:shape=(2,num_layers), each element of the list is a tuple (h, c) for hidden state and memory
layer_output_list:the list of lists of length T of each output
    单层列表，每个元素表示一层LSTM层的输出状态h
    h.size=[B,T,hidden_dim,H,W]
last_state_list:the list of last states
    双层列表，每个元素表示每一层的最后一个timestep的输出状态[h,c]
    h.size=c.size=[B,hidden_dim,H,W]
----------
Example
----------
>> x = torch.rand((B=32, T=10, C=64, H=128, W=128))
>> convlstm = ConvLSTM(in=64, hid=16, k=3, num=1, True, True, False)
>> _, last_states = convlstm(x)
>> h = last_states[0][0]  # 0 for layer index, 0 for h index
"""
class ConvLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, kernel_size, num_layers, seq_len,
                 batch_first=False, bias=True, return_all_layers=False):
        """
        Initialize ConvLSTM Network.
        ----------
        Parameters
        ----------
        input_dim: Number of channels in input 输入张量x的通道数
        hidden_dim: Number of hidden channels 状态张量h,c的通道数，可以是一个列表
        output_dim: Number of channels in output 输出张量y的通道数
        seq_len: (input sequence length, output sequence length) out_len should be shorter than in_len
        kernel_size: Size of kernel in convolutions 卷积核的尺寸，默认所有层的卷积核尺寸都是一样的
        num_layers: Number of LSTM layers stacked on each other 堆砌LSTM单元层数，与len(hidden_dim)相等
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or not in Convolution
        return_all_layers: Return the list of computations for all layers 是否返回所有LSTM层的h状态
        [Note] Will do same padding 相同的卷积核尺寸，相同的padding尺寸
        """
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)  # 转为列表
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)  # 转为列表
        if not len(kernel_size) == len(hidden_dim) == num_layers:  # 判断一致性
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_all_layers = return_all_layers
        self.in_seqLen = seq_len[0]
        self.out_seqLen = seq_len[1]

        # 多层LSTM设置,简单按层数堆叠
        cell_list = []
        for i in range(0, self.num_layers):
            # 当前LSTM层的输入维度
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMBlock(input_dim=cur_input_dim,
                                           hidden_dim=self.hidden_dim[i],
                                           kernel_size=self.kernel_size[i],
                                           bias=self.bias))
        # 把定义的多个LSTM层串联成网络模型
        self.cell_list = nn.ModuleList(cell_list)
        # h(L)通过conv3d得到最终输出y_hat
        # k_T = self.in_seqLen - self.out_seqLen + 3
        self.conv3d = nn.Conv3d(in_channels=self.hidden_dim[-1],
                                out_channels=self.output_dim,
                                kernel_size=(3, 3, 3),
                                stride=(1, 1, 1),
                                padding=(1, 1, 1))
        self.proj = nn.Linear(self.in_seqLen, self.out_seqLen, bias=True)

    def forward(self, input_tensor, A, x_timeF, y_timeF, hidden_state=None):
        """
        Forward Propagation.
        ----------
        Parameters
        ----------
        input_tensor: 5-D Tensor x_t either of shape [B, T, C, H, W] or [T, B, C, H, W]
        hidden_state:
        ----------
        Returns
        ----------
        last_state_list, layer_output
        """
        # [B, T, H, W] -> [B, T, C, H, W]
        input_tensor = input_tensor.unsqueeze(2)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # 获取 b,h,w信息
            b, _, _, h, w = input_tensor.size()
            # init hidden state
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        #last_state_list = []

        # 根据输入张量获取lstm的长度 seq_len=T
        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        # 逐层计算
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            # 逐timestep计算
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                # 第layer_idx层的第t个timestep的输出状态
                output_inner.append(h)
            # 串联第layer_idx层的所有timestep的输出状态
            layer_output = torch.stack(output_inner, dim=1)
            # 准备第layer_idx+1层的输入张量
            cur_layer_input = layer_output
            # 串联当前层的所有timestep的状态h
            layer_output_list.append(layer_output)
            # 当前层的最后一个timestep的输出状态[h,c]
            #last_state_list.append([h, c])
        # 取得最后LSTM层的输出 [B,T,hidden_dim,H,W]
        H = layer_output_list[-1:][0]
        # 进行一次3dConv得到结果y_(t+h:t+h+window_size)
        # [B,T,hidden_dim,H,W]->[B,hidden_dim,T,H,W]
        H = H.permute(0, 2, 1, 3, 4)
        # [B,hidden_dim,T_in,H,W]->[B, output_dim, T_in, H, W]
        output_tensor=self.conv3d(H)
        # [B, output_dim, T_in, H, W]->[B, output_dim, H, W, T_in]
        output_tensor = output_tensor.permute(0, 1, 3, 4, 2)
        # [B, output_dim, H, W, T_in] -->[B, output_dim, H, W, T_out]
        output_tensor = self.proj(output_tensor)
        # [B, output_dim, H, W, T_out] -->[B, T_out, output_dim, H, W]
        output_tensor = output_tensor.permute(0, 4, 1, 2, 3)
        # [B, T_out, output_dim=1, H, W] -->[B, T_out, H, W]
        output_tensor = output_tensor.squeeze(2)

        return output_tensor

    def _init_hidden(self, batch_size, image_size):
        """
        初始化所有lstm层的第一个timestep的输入状态h_0,c_0
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        检测输入的kernel_size是否符合要求，要求kernel_size的格式是list或tuple
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        扩展到多层lstm情况
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


if __name__ == "__main__":
    # load data
    data = torch.randn((5, 6, 3, 30, 30))
    # build model
    model = ConvLSTM(input_dim=3,
                     hidden_dim=[32, 32, 64],
                     output_dim=3,
                     kernel_size=[(3, 3), (5, 5), (7, 7)],
                     num_layers=3,
                     seq_len=(6, 6),
                     batch_first=True,
                     bias=True,
                     return_all_layers=True)
    # model forward
    pred = model(data)
    print(pred.shape)

