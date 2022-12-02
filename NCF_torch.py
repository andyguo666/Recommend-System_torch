# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Author    : guomiansheng
@Software  : Pycharm
@Contact   : 864934027@qq.com
@File      : NCF_torch.py
@Time      : 2022/12/2 17:14
"""
# import paddle
import torch
# from paddle import nn
from torch import nn
# from paddle.io import DataLoader, Dataset
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import copy
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score,log_loss
from tqdm import tqdm
from collections import defaultdict
import math
import random
import warnings
warnings.filterwarnings("ignore")

#参数配置

config = {
    'train_path':'./data/data173799/train_enc.csv',
    'valid_path':'./data/data173799/valid_enc.csv',
    'test_path':'./data/data173799/test_enc.csv',
    "debug_mode" : True,
    "epoch" : 10,
    "batch" : 20480,
    "lr" : 0.001,
}


# 训练集太大了，一共有900w条数据，这里用valid数据来完成流程
df = pd.read_csv(config['valid_path'])
# 统计每个用户的记录出现次数，作为字段user_count
df['user_count'] = df['user_id'].map(df['user_id'].value_counts())
# 筛选出用户出现次数超过20的用户，并且重置索引
df = df[df['user_count']>20].reset_index(drop=True)
# 将每个用户的记录组成序列，并且将所有用户组成字典
pos_dict = df.groupby('user_id')['item_id'].apply(list).to_dict()

'''
这里的样本构造逻辑是来自于《Neural Collaborative Filtering》的实验部分的样本构造逻辑
'''
# 负采样
ratio = 3
# 构造样本
train_user_list = []
train_item_list = []
train_label_list = []

test_user_list = []
test_item_list = []
test_label_list = []
if config['debug_mode']:
    user_list = df['user_id'].unique()[:100]
else:
    user_list = df['user_id'].unique()

item_list = df['item_id'].unique()
item_num = df['item_id'].nunique()


for user in tqdm(user_list):
    # 训练集正样本
    for i in range(len(pos_dict[user]) - 1):
        train_user_list.append(user)
        train_item_list.append(pos_dict[user][i])
        train_label_list.append(1)

    # 测试集正样本
    test_user_list.append(user)
    test_item_list.append(pos_dict[user][-1])
    test_label_list.append(1)

    # 训练集：每个用户负样本数
    user_count = len(pos_dict[user]) - 1  # 训练集 用户行为序列长度
    neg_sample_per_user = user_count * ratio

    for i in range(neg_sample_per_user):
        train_user_list.append(user)
        temp_item_index = random.randint(0, item_num - 1)
        # 为了防止 负采样选出来的Item 在用户的正向历史行为序列(pos_dict)当中
        while item_list[temp_item_index] in pos_dict[user]:
            temp_item_index = random.randint(0, item_num - 1)
        train_item_list.append(item_list[temp_item_index])
        train_label_list.append(0)

    # 测试集合：每个用户负样本数为 100(论文设定)
    for i in range(100):
        test_user_list.append(user)
        temp_item_index = random.randint(0, item_num - 1)
        # 为了防止 负采样选出来的Item 在用户的正向历史行为序列(pos_dict)当中
        while item_list[temp_item_index] in pos_dict[user]:
            temp_item_index = random.randint(0, item_num - 1)
        test_item_list.append(item_list[temp_item_index])
        test_label_list.append(0)


train_df = pd.DataFrame()
train_df['user_id'] = train_user_list
train_df['item_id'] = train_item_list
train_df['label'] = train_label_list

test_df = pd.DataFrame()
test_df['user_id'] = test_user_list
test_df['item_id'] = test_item_list
test_df['label'] = test_label_list


vocab_map = {
    'user_id':df['user_id'].max()+1,
    'item_id':df['item_id'].max()+1
}


# Dataset构造
class BaseDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.feature_name = ['user_id', 'item_id']
        # 数据编码
        self.enc_data()

    def enc_data(self):
        # 使用enc_dict对数据进行编码
        self.enc_data = defaultdict(dict)
        # defaultdict中当key不存在且查找时，会返回默认值
        for col in self.feature_name:
            self.enc_data[col] = torch.Tensor(np.array(self.df[col])).squeeze(-1)

    def __getitem__(self, index):
        data = dict()
        for col in self.feature_name:
            data[col] = self.enc_data[col][index]
        if 'label' in self.df.columns:
            # data['label'] = torch.Tensor([self.df['label'].iloc[index]], dtype="float32").squeeze(-1)
            data['label'] = torch.tensor([self.df['label'].iloc[index]], dtype=torch.float32).squeeze(-1)
        return data

    def __len__(self):
        return len(self.df)


train_dataset = BaseDataset(train_df)
test_dataset = BaseDataset(test_df)


'''
model: NCF
'''


class NCF(nn.Module):
    def __init__(self,
                 embedding_dim=16,
                 vocab_map=None,
                 loss_fun='nn.BCELoss()'):
        super(NCF, self).__init__()
        self.embedding_dim = embedding_dim
        self.vocab_map = vocab_map
        self.loss_fun = eval(loss_fun)  # self.loss_fun  = paddle.nn.BCELoss()

        self.user_emb_layer = nn.Embedding(self.vocab_map['user_id'],
                                           self.embedding_dim)
        self.item_emb_layer = nn.Embedding(self.vocab_map['item_id'],
                                           self.embedding_dim)

        self.mlp = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data):
        data['user_id'] = data['user_id'].long()
        data['item_id'] = data['item_id'].long()
        user_emb = self.user_emb_layer(data['user_id'])  # [batch,emb]
        item_emb = self.item_emb_layer(data['item_id'])  # [batch,emb]
        # 水平方向进行拼接concat,axis维度写1也可以
        mlp_input = torch.concat([user_emb, item_emb], axis=-1).squeeze(1)
        y_pred = self.mlp(mlp_input)
        if 'label' in data.keys():
            loss = self.loss_fun(y_pred.squeeze(), data['label'])
            output_dict = {'pred': y_pred, 'loss': loss}
        else: # test/valid
            output_dict = {'pred': y_pred}
        return output_dict



#训练模型，验证模型
def train_model(model, train_loader, optimizer, metric_list=['roc_auc_score','log_loss']):
    model.train()
    pred_list = []
    label_list = []
    pbar = tqdm(train_loader)
    for data in pbar:

        output = model(data)
        pred = output['pred']
        loss = output['loss']

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())
        # pbar.set_description("Loss {}".format(loss.numpy()[0]))
        loss = loss.cpu().detach()
        pbar.set_description("Loss {}".format(loss.numpy()))

    res_dict = dict()
    for metric in metric_list:
        if metric =='log_loss':
            res_dict[metric] = log_loss(label_list,pred_list, eps=1e-7)
        else:
            res_dict[metric] = eval(metric)(label_list,pred_list)

    return res_dict


def valid_model(model, valid_loader, metric_list=['roc_auc_score','log_loss']):
    model.eval()
    pred_list = []
    label_list = []

    for data in (valid_loader):

        output = model(data)
        pred = output['pred']

        pred_list.extend(pred.squeeze(-1).cpu().detach().numpy())
        label_list.extend(data['label'].squeeze(-1).cpu().detach().numpy())

    res_dict = dict()
    for metric in metric_list:
        if metric =='log_loss':
            res_dict[metric] = log_loss(label_list,pred_list, eps=1e-7)
        else:
            res_dict[metric] = eval(metric)(label_list,pred_list)

    return res_dict


def test_model(model, test_loader):
    model.eval()
    pred_list = []

    for data in tqdm(test_loader):

        output = model(data)
        pred = output['pred']
        pred_list.extend(pred.squeeze().cpu().detach().numpy())

    return np.array(pred_list)


#dataloader
train_loader = DataLoader(train_dataset,batch_size=config['batch'],shuffle=True,num_workers=0)
test_loader = DataLoader(test_dataset,batch_size=config['batch'],shuffle=False,num_workers=0)


model = NCF(embedding_dim=64,vocab_map=vocab_map)
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])
train_metric_list = []
#模型训练流程
for i in range(config['epoch']):
    #模型训练
    train_metirc = train_model(model,train_loader,optimizer=optimizer)
    train_metric_list.append(train_metirc)

    print("Train Metric:")
    print(train_metirc)



# 计算指标
y_pre = test_model(model,test_loader)
test_df['y_pre'] = y_pre
test_df['ranking'] = test_df.groupby(['user_id'])['y_pre'].rank(method='first', ascending=False)
test_df = test_df.sort_values(by=['user_id','ranking'],ascending=True)
test_df


def hitrate(test_df,k=20):
    user_num = test_df['user_id'].nunique()
    test_gd_df = test_df[test_df['ranking']<=k].reset_index(drop=True)
    return test_gd_df['label'].sum() / user_num
print("hitrate result:\n", hitrate(test_df,k=5))


def ndcg(test_df, k=20):
    '''
    idcg@k 一定为1
    dcg@k 1/log_2(ranking+1) -> log(2)/log(ranking+1)
    '''
    user_num = test_df['user_id'].nunique()
    test_gd_df = test_df[test_df['ranking'] <= k].reset_index(drop=True)

    test_gd_df = test_gd_df[test_gd_df['label'] == 1].reset_index(drop=True)
    test_gd_df['ndcg'] = math.log(2) / np.log(test_gd_df['ranking'] + 1)
    return test_gd_df['ndcg'].sum() / user_num
print("ndcg result:\n", ndcg(test_df,k=5))


def plot_metric(metric_dict_list, metric_name):
    epoch_list = [x for x in range(1,1+len(metric_dict_list))]
    metric_list = [metric_dict_list[i][metric_name] for i in range(len(metric_dict_list))]
    plt.figure(dpi=100)
    plt.plot(epoch_list,metric_list)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name)
    plt.title('Train Metric')
    plt.show()


plot_metric(train_metric_list,'roc_auc_score')
plot_metric(train_metric_list,'log_loss')

