import numpy as np
import torch
from torch.utils import  data
from d2l import torch as d2l
from torch import nn

#生成数据集
true_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

#读取数据集
def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays) #这里*加变量名是python中的拆包操作
    return data.DataLoader(dataset,batch_size,shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# a=next(iter(data_iter))
# print(a)

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)


