# !/usr/bin/python
# -*- coding: utf-8 -*-
"""
@Author    : guomiansheng
@Software  : Pycharm
@Contact   : 864934027@qq.com
@File      : MIND.py
@Time      : 2022/11/24 16:04
"""
# import paddle
import torch
# from paddle import nn
from torch import nn
# from paddle.io import DataLoader, Dataset
from torch.utils.data import Dataset, DataLoader, random_split
# import paddle.nn.functional as F
import torch.nn.functional as F
import pandas as pd
import numpy as np
import copy
import os
import math
import random
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.preprocessing import normalize
from tqdm import tqdm
from collections import defaultdict
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import faiss
from torch.nn import Parameter
# torch.cuda.set_device('cuda:0')
# paddle.device.set_device('gpu:0')
# import warnings
# warnings.filterwarnings("ignore")


class SeqnenceDataset(Dataset):
    def __init__(self, config, df, phase='train'):
        self.config = config
        self.df = df
        self.max_length = self.config['max_length']
        self.df = self.df.sort_values(by=['user_id', 'timestamp'])
        self.user2item = self.df.groupby('user_id')['item_id'].apply(list).to_dict()
        self.user_list = self.df['user_id'].unique()
        self.phase = phase

    def __len__(self, ):
        return len(self.user2item)

    def __getitem__(self, index):
        if self.phase == 'train':
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = random.choice(range(4, len(item_list)))  # 从[8,len(item_list))中随机选择一个index
            # k = np.random.randint(2,len(item_list))
            item_id = item_list[k]  # 该index对应的item加入item_id_list

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return torch.Tensor(hist_item_list).squeeze(0), torch.Tensor(hist_mask_list).squeeze(
                0), torch.Tensor([item_id])
        else:
            user_id = self.user_list[index]
            item_list = self.user2item[user_id]
            hist_item_list = []
            hist_mask_list = []

            k = int(0.8 * len(item_list))
            # k = len(item_list)-1

            if k >= self.max_length:  # 选取seq_len个物品
                hist_item_list.append(item_list[k - self.max_length: k])
                hist_mask_list.append([1.0] * self.max_length)
            else:
                hist_item_list.append(item_list[:k] + [0] * (self.max_length - k))
                hist_mask_list.append([1.0] * k + [0.0] * (self.max_length - k))

            return torch.Tensor(hist_item_list).squeeze(0), torch.Tensor(hist_mask_list).squeeze(
                0), item_list[k:]

    def get_test_gd(self):
        self.test_gd = {}
        for user in self.user2item:
            item_list = self.user2item[user]
            test_item_index = int(0.8 * len(item_list))
            self.test_gd[user] = item_list[test_item_index:]
        return self.test_gd


class CapsuleNetwork(nn.Module):

    def __init__(self, hidden_size, seq_len, bilinear_type=2, interest_num=4, routing_times=3, hard_readout=True,
                 relu_layer=False):
        super(CapsuleNetwork, self).__init__()
        self.hidden_size = hidden_size  # h
        self.seq_len = seq_len  # s
        self.bilinear_type = bilinear_type
        # 兴趣个数
        self.interest_num = interest_num
        self.routing_times = routing_times
        self.hard_readout = hard_readout
        self.relu_layer = relu_layer
        self.stop_grad = True
        self.relu = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.ReLU()
        )
        if self.bilinear_type == 0:  # MIND
            self.linear = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        elif self.bilinear_type == 1:
            self.linear = nn.Linear(self.hidden_size, self.hidden_size * self.interest_num, bias=False)
        else:  # ComiRec_DR
            self.w = self.create_parameter(
                shape=[1, self.seq_len, self.interest_num * self.hidden_size, self.hidden_size])

    def forward(self, item_eb, mask):
        if self.bilinear_type == 0:  # MIND
            item_eb_hat = self.linear(item_eb)  # [b, s, h]
            item_eb_hat = torch.repeat_interleave(item_eb_hat, self.interest_num, 2) # [b, s, h*in]
        elif self.bilinear_type == 1:
            item_eb_hat = self.linear(item_eb)
        else:  # ComiRec_DR
            # shape=(batch_size, maxlen, 1, embedding_dim)
            u = torch.unsqueeze(item_eb, 2)
            item_eb_hat = torch.sum(self.w[:, :self.seq_len, :, :] * u,
                                    3)  # shape=(batch_size, maxlen, hidden_size*interest_num)

        item_eb_hat = torch.reshape(item_eb_hat, (-1, self.seq_len, self.interest_num, self.hidden_size))
        item_eb_hat = torch.transpose(item_eb_hat, 2, 1)
        # item_eb_hat = paddle.reshape(item_eb_hat, (-1, self.interest_num, self.seq_len, self.hidden_size))

        # [b, in, s, h]
        if self.stop_grad:  # 截断反向传播，item_emb_hat不计入梯度计算中
            item_eb_hat_iter = item_eb_hat.detach()
        else:
            item_eb_hat_iter = item_eb_hat

        # b的shape=(b, in, s)
        if self.bilinear_type > 0:  # b初始化为0（一般的胶囊网络算法）
            capsule_weight = torch.zeros((item_eb_hat.shape[0], self.interest_num, self.seq_len))
        else:  # MIND使用高斯分布随机初始化b
            capsule_weight = torch.randn((item_eb_hat.shape[0], self.interest_num, self.seq_len))

        # 动态路由传播3次
        for i in range(self.routing_times):
            atten_mask = torch.repeat_interleave(torch.unsqueeze(mask, 1), self.interest_num, 1) # [b, in, s]
            paddings = torch.zeros_like(atten_mask)

            # 计算c，进行mask，最后shape=[b, in, 1, s]
            capsule_softmax_weight = F.softmax(capsule_weight)
            # capsule_softmax_weight = F.softmax(capsule_weight, axis=-1)
            capsule_softmax_weight = torch.where(atten_mask==0, paddings, capsule_softmax_weight)  # mask
            capsule_softmax_weight = torch.unsqueeze(capsule_softmax_weight, 2)

            if i < 2:
                # s=c*u_hat , (batch_size, interest_num, 1, seq_len) * (batch_size, interest_num, seq_len, hidden_size)
                interest_capsule = torch.matmul(capsule_softmax_weight,
                                                item_eb_hat_iter)  # shape=(batch_size, interest_num, 1, hidden_size)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)  # shape=(batch_size, interest_num, 1, 1)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)  # shape同上
                interest_capsule = scalar_factor * interest_capsule  # squash(s)->v,shape=(batch_size, interest_num, 1, hidden_size)

                # 更新b
                delta_weight = torch.matmul(item_eb_hat_iter,  # shape=(batch_size, interest_num, seq_len, hidden_size)
                                            torch.transpose(interest_capsule, 3, 2)
                                            # shape=(batch_size, interest_num, hidden_size, 1)
                                            )  # u_hat*v, shape=(batch_size, interest_num, seq_len, 1)
                delta_weight = torch.reshape(delta_weight, (
                -1, self.interest_num, self.seq_len))  # shape=(batch_size, interest_num, seq_len)
                capsule_weight = capsule_weight + delta_weight  # 更新b
            else:
                interest_capsule = torch.matmul(capsule_softmax_weight, item_eb_hat)
                cap_norm = torch.sum(torch.square(interest_capsule), -1, keepdim=True)
                scalar_factor = cap_norm / (1 + cap_norm) / torch.sqrt(cap_norm + 1e-9)
                interest_capsule = scalar_factor * interest_capsule

        interest_capsule = torch.reshape(interest_capsule, (-1, self.interest_num, self.hidden_size))

        # MIND模型使用book数据库时，使用relu_layer
        if self.relu_layer:
            interest_capsule = self.relu(interest_capsule)

        return interest_capsule


class MIND(nn.Module):
    def __init__(self, config):
        super(MIND, self).__init__()

        self.config = config
        self.embedding_dim = self.config['embedding_dim']
        self.max_length = self.config['max_length']
        self.n_items = self.config['n_items']

        self.item_emb = nn.Embedding(self.n_items, self.embedding_dim, padding_idx=0)
        # capsule network
        self.capsule = CapsuleNetwork(self.embedding_dim, self.max_length, bilinear_type=0,
                                      interest_num=self.config['K'])
        self.loss_fun = nn.CrossEntropyLoss()
        self.reset_parameters()

    def calculate_loss(self,user_emb,pos_item):
        all_items = self.item_emb.weight
        scores = torch.matmul(user_emb, all_items.transpose(1, 0))
        pos_item = pos_item.squeeze(1).long()
        return self.loss_fun(scores,pos_item)

    def output_items(self):
        return self.item_emb.weight


    def reset_parameters(self, initializer=None):
        for weight in self.parameters():
            if len(weight.shape) < 2:
                torch.nn.init.kaiming_normal_(weight.unsqueeze(0))
            else:
                torch.nn.init.kaiming_normal_(weight)


    def forward(self, item_seq, mask, item, train=True):

        if train:
            # 1. embedding layer
            item_seq = item_seq.long()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            item_e = self.item_emb(item.long()).squeeze(1)

            # 2. multi-interest extractor layer + 3. label-aware attention layer
            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb
            cos_res = torch.bmm(multi_interest_emb, item_e.squeeze(1).unsqueeze(-1))
            # 取内积结果最大的，作为最后的得分，并且取出对应的index
            k_index = torch.argmax(cos_res, axis=1)
            best_interest_emb = torch.rand((multi_interest_emb.shape[0], multi_interest_emb.shape[2]))
            for k in range(multi_interest_emb.shape[0]):
                best_interest_emb[k, :] = multi_interest_emb[k, k_index[k], :]

            # 4. loss function
            loss = self.calculate_loss(best_interest_emb,item)
            output_dict = {
                'user_emb': multi_interest_emb,
                'loss': loss,
            }
        else:
            # test stage
            item_seq = item_seq.long()
            seq_emb = self.item_emb(item_seq)  # Batch,Seq,Emb
            multi_interest_emb = self.capsule(seq_emb, mask)  # Batch,K,Emb
            output_dict = {
                'user_emb': multi_interest_emb,
            }
        return output_dict



config = {
    'train_path':'./data/data173799/train_enc.csv',
    'valid_path':'./data/data173799/valid_enc.csv',
    'test_path':'./data/data173799/test_enc.csv',
    'lr':1e-4,
    'Epoch':5,
    'batch_size':256,
    'embedding_dim':16,
    'num_layers':1,
    'max_length':20,
    'n_items':15406,
    'K':4
}


def my_collate(batch):
    hist_item, hist_mask, item_list = list(zip(*batch))

    hist_item = [x.unsqueeze(0) for x in hist_item]
    hist_mask = [x.unsqueeze(0) for x in hist_mask]

    hist_item = torch.cat(hist_item,axis=0)
    hist_mask = torch.cat(hist_mask,axis=0)
    return hist_item,hist_mask,item_list


def save_model(model, path):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(model.state_dict(), path + 'model.pth')


def load_model(model, path):
    state_dict = torch.load(path + 'model.pth')
    model.load_state_dict(state_dict)
    print('model loaded from %s' % path)
    return model



'''
note: 基于faiss的向量召回
'''
def get_predict(model, test_data, hidden_size, topN=20):
    item_embs = model.output_items().cpu().detach().numpy()
    item_embs = normalize(item_embs, norm='l2')
    gpu_index = faiss.IndexFlatIP(hidden_size)
    gpu_index.add(item_embs)

    test_gd = dict()
    preds = dict()

    user_id = 0

    for (item_seq, mask, targets) in tqdm(test_data):

        # 获取用户嵌入
        # 多兴趣模型，shape=(batch_size, num_interest, embedding_dim)
        # 其他模型，shape=(batch_size, embedding_dim)
        user_embs = model(item_seq, mask, None, train=False)['user_emb']
        user_embs = user_embs.cpu().detach().numpy()

        # 用内积来近邻搜索，实际是内积的值越大，向量越近（越相似）
        if len(user_embs.shape) == 2:  # 非多兴趣模型评估
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            D, I = gpu_index.search(user_embs, topN)  # Inner Product近邻搜索，D为distance，I是index
            #             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)
            for i, iid_list in enumerate(targets):  # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
                test_gd[user_id] = iid_list
                preds[user_id] = I[i, :]
                user_id += 1
        else:  # 多兴趣模型评估
            ni = user_embs.shape[1]  # num_interest
            user_embs = np.reshape(user_embs,
                                   [-1, user_embs.shape[-1]])  # shape=(batch_size*num_interest, embedding_dim)
            user_embs = normalize(user_embs, norm='l2').astype('float32')
            # Inner Product近邻搜索，D为distance，I是index
            D, I = gpu_index.search(user_embs, topN)
            #             D,I = faiss.knn(user_embs, item_embs, topN,metric=faiss.METRIC_INNER_PRODUCT)

            # 每个用户的label列表，此处item_id为一个二维list，验证和测试是多label的
            for i, iid_list in enumerate(targets):
                recall = 0
                dcg = 0.0
                item_list_set = []

                # 将num_interest个兴趣向量的所有topN近邻物品（num_interest*topN个物品）集合起来按照距离重新排序
                item_list = list(
                    zip(np.reshape(I[i * ni:(i + 1) * ni], -1), np.reshape(D[i * ni:(i + 1) * ni], -1)))
                # 降序排序，内积越大，向量越近
                item_list.sort(key=lambda x: x[1], reverse=True)
                # 按距离由近到远遍历推荐物品列表，最后选出最近的topN个物品作为最终的推荐物品
                for j in range(len(item_list)):
                    if item_list[j][0] not in item_list_set and item_list[j][0] != 0:
                        item_list_set.append(item_list[j][0])
                        if len(item_list_set) >= topN:
                            break
                test_gd[user_id] = iid_list
                preds[user_id] = item_list_set
                user_id += 1
    return test_gd, preds


def evaluate(preds, test_gd, topN=50):
    total_recall = 0.0
    total_ndcg = 0.0
    total_hitrate = 0
    for user in test_gd.keys():
        recall = 0
        dcg = 0.0
        item_list = test_gd[user]
        for no, item_id in enumerate(item_list):
            if item_id in preds[user][:topN]:
                recall += 1
                dcg += 1.0 / math.log(no + 2, 2)
            idcg = 0.0
            for no in range(recall):
                idcg += 1.0 / math.log(no + 2, 2)
        total_recall += recall * 1.0 / len(item_list)
        if recall > 0:
            total_ndcg += dcg / idcg
            total_hitrate += 1
    total = len(test_gd)
    recall = total_recall / total
    ndcg = total_ndcg / total
    hitrate = total_hitrate * 1.0 / total
    return {f'recall@{topN}': recall, f'ndcg@{topN}': ndcg, f'hitrate@{topN}': hitrate}


# 指标计算
def evaluate_model(model, test_loader, embedding_dim, topN=20):
    test_gd, preds = get_predict(model, test_loader, embedding_dim, topN=topN)
    return evaluate(preds, test_gd, topN=topN)


# 读取数据
train_df = pd.read_csv(config['train_path'])
valid_df = pd.read_csv(config['valid_path'])
test_df = pd.read_csv(config['test_path'])
train_dataset = SeqnenceDataset(config, train_df, phase='train')
valid_dataset = SeqnenceDataset(config, valid_df, phase='test')
test_dataset = SeqnenceDataset(config, test_df, phase='test')
train_loader = DataLoader(dataset=train_dataset, batch_size=config['batch_size'], shuffle=True,num_workers=8)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=config['batch_size'], shuffle=False,collate_fn=my_collate)
test_loader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'], shuffle=False,collate_fn=my_collate)


model = MIND(config)
# model = SRGNN(config)
optimizer = torch.optim.Adam(params=model.parameters(), lr=config['lr'])
# optimizer = torch.optimizer.Adam(parameters=model.parameters(), learning_rate=config['lr'])
log_df = pd.DataFrame()
best_reacall = -1

exp_path = './ml-20m_softmax/MIND_{}_{}_{}/'.format(config['lr'],config['batch_size'],config['embedding_dim'])
os.makedirs(exp_path,exist_ok=True,mode=0o777)
patience = 5
last_improve_epoch = 1
log_csv = exp_path+'log.csv'
# *****************************************************train*********************************************
for epoch in range(1, 1 + config['Epoch']):
    # try :
        pbar = tqdm(train_loader)
        model.train()
        loss_list = []
        acc_50_list = []
        print()
        print('Training:')
        print()
        for batch_data in pbar:
            (item_seq, mask, item) = batch_data

            output_dict = model(item_seq, mask, item)
            loss = output_dict['loss']

            loss.backward()
            optimizer.step()
            # optimizer.clear_grad()
            optimizer.zero_grad()

            loss_list.append(loss.item())

            pbar.set_description('Epoch [{}/{}]'.format(epoch,config['Epoch']))
            pbar.set_postfix(loss = np.mean(loss_list))
    # *****************************************************valid*********************************************

        print('Valid')
        recall_metric = evaluate_model(model, valid_loader, config['embedding_dim'], topN=50)
        print(recall_metric)
        recall_metric['phase'] = 'valid'
        log_df = log_df.append(recall_metric, ignore_index=True)
        log_df.to_csv(log_csv)

        if recall_metric['recall@50'] > best_reacall:
            save_model(model,exp_path)
            best_reacall = recall_metric['recall@50']
            last_improve_epoch = epoch

        if epoch - last_improve_epoch > patience:
            break

print('Testing')
model = load_model(model,exp_path)
recall_metric = evaluate_model(model, test_loader, config['embedding_dim'], topN=50)
print(recall_metric)
recall_metric['phase'] = 'test'
log_df = log_df.append(recall_metric, ignore_index=True)
log_df.to_csv(log_csv)


# embedding分布可视化
def plot_embedding(data, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure(dpi=120)
    plt.scatter(data[:, 0], data[:, 1], marker='.')

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()


item_emb = model.output_items().detach().numpy()
tsne_emb = TSNE(n_components=2).fit_transform(item_emb)
plot_embedding(tsne_emb,'MIND Item Embedding')