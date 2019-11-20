# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:47:58 2019

@author: yorksywang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:35:00 2019

@author: yorksywang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 11:43:47 2019

@author: yorksywang
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 11:27:42 2019

@author: yorksywang
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 14:52:52 2019

@author: yorksywang
"""

import sklearn.svm as svm 
import matplotlib.pyplot as plt
import numpy as np
import torch
from functools import reduce
import torch.optim as optim
import torch.nn as nn
from torch.nn import utils
from torch.nn import init
#from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing  import StandardScaler
#from model import LinearSVM
import pandas as pd
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
import random
import torch.nn.functional as F
import argparse

global_input_size=256


def same_class_augmentation(x_to_aug,y_to_aug):
    """ Perform same class augmentation of the wave by loading a random segment
    from the class_dir and additively combine the wave with that segment.
    """
    #sig_paths = glob.glob(os.path.join(class_dir, "*.wav"))
    #aug_sig_path = np.random.choice(sig_paths, 1, replace=False)[0]
    #(fs, aug_sig) = utils.read_wave_file(aug_sig_path)
    #print(np.shape(x_to_aug))
    output_x=x_to_aug
    output_y=np.concatenate((y_to_aug,y_to_aug),axis=0)
    for i in range(np.shape(x_to_aug)[0]):
        x_temp1 =np.mean(random.choices(x_to_aug,k=5),0)
        alpha = np.random.rand()
        to_add=(1.0-alpha)*x_to_aug[i,:] + alpha*x_temp1
        #print(np.shape(to_add))
        output_x=np.concatenate( (output_x,to_add.reshape(1,global_input_size)),axis=0)
    #print(np.shape(output_x))
    return output_x,output_y


def get_data(test_size=0.2):
    #file_name=['word_list_gamble_20190927','training_data_weg','training_data_gam','training_data_bag']
   # i=0
    #df_train=pd.read_csv('./data/'+file_name[i]+'.csv',encoding='utf-8',header=1)
    df_train=pd.read_csv('../nlp/training_data_20191119.csv',encoding='utf-8',header=0,index_col=0)
    df_word=pd.read_csv('../nlp/training_data_20191119_word2vec_dict.csv',encoding='utf-8',header=0,index_col=0)
    
    x_train=np.array(df_train.values[:,0:global_input_size]).astype('float64') 
    y_train=np.array(df_train.values[:,global_input_size]).astype('int')
    z_train=np.array(df_word.values[:]).astype('str')

    y=y_train[np.where(y_train>=0)]
    x=x_train[np.where(y_train>=0)]
    #X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.15,random_state=20170816)

    y_temp=y_train[np.where(y_train>=1)]
    x_temp=x_train[np.where(y_train>=1)]
    x_temp,y_temp=same_class_augmentation(x_temp,y_temp)
    x_temp,y_temp=same_class_augmentation(x_temp,y_temp)
    x_temp,y_temp=same_class_augmentation(x_temp,y_temp)
    x_temp,y_temp=same_class_augmentation(x_temp,y_temp)
    x_temp,y_temp=same_class_augmentation(x_temp,y_temp)
    
    #x_temp,y_temp=same_class_augmentation(x_temp,y_temp)
    print(x_temp.shape)
    y=np.concatenate((y,y_temp),axis=0)
    x=np.concatenate((x,x_temp),axis=0)
    y_temp=y_train[np.where(y_train==0)]
    x_temp=x_train[np.where(y_train==0)]
    print(x_temp.shape)
    x_temp,y_temp=same_class_augmentation(x_temp,y_temp)
    
    print(x_temp.shape)
    y=np.concatenate((y,y_temp),axis=0)
    x=np.concatenate((x,x_temp),axis=0)
# =============================================================================
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     y=np.concatenate((y,y_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
#     x=np.concatenate((x,x_temp),axis=0)
# =============================================================================
    ran_index=np.random.permutation(np.shape(y)[0])
    
    y=y[ran_index]
    x=x[ran_index]
    
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=test_size,random_state=20170816)
    #print(X_train.shape)

# =============================================================================
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     Y_test=np.concatenate((Y_test,y_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
#     X_test=np.concatenate((X_test,x_test_temp),axis=0)
# =============================================================================
    
    sc=StandardScaler()
    
    sc.fit(X_train)
    
    X_train_nor=sc.transform(X_train)
    X_test_nor=sc.transform(X_test)
    
    z_train=z_train[np.where(y_train<0)]
    x_grey=x_train[np.where(y_train<0)]
    X_grey_nor=sc.transform(x_grey)
    return X_train_nor, Y_train,X_test_nor,Y_test,X_grey_nor,z_train

class Classfier(torch.nn.Module):
    def __init__(self,args):
        super(Classfier, self).__init__()
        self.layer1 = torch.nn.Conv1d(1,10,3)
        #self.act1 = torch.nn.ReLU()
        #self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
        self.layer2 = torch.nn.Conv1d(10,10,3)
        self.maxpool1=torch.nn.MaxPool1d(2)
        self.flatten = torch.nn.Flatten()
        self.layer3 = torch.nn.Linear((global_input_size-4)*5,1)
        #self.model.to(device)
        #self.epoches = epoches
        
    def forward(self, x):
        #x = self.drop_out1(x)
        x = F.relu(self.layer1(x))
        #print(x.shape)
        x = F.relu(self.layer2(x))
        #print(x.shape)
        x = self.maxpool1(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        x = self.layer3(x)
       # print(x.shape)


        return x
# =============================================================================
# class Classfier(torch.nn.Module):
#     def __init__(self):
#         super(Classfier, self).__init__()
#         self.layer1 = torch.nn.Linear(128,32)
#         self.layer2 = torch.nn.Linear(32,1)
#         #self.model.to(device)
#         #self.epoches = epoches
#         
#     def forward(self, x):
#         #x = self.drop_out1(x)
#         x = F.relu(self.layer1(x))
#         x = self.layer2(x)
#        # print(x.shape)
# 
# 
#         return x
# =============================================================================

# =============================================================================
# class Classfier(torch.nn.Module):
#     def __init__(self,args):
#         super(Classfier, self).__init__()
#         self.layer1 = torch.nn.Linear(args.input_size,64)
#         self.layer2 = torch.nn.Linear(64,32)
#         self.layer3 = torch.nn.Linear(32,args.output_size)
#         torch.nn.init.xavier_uniform_(self.layer1.weight)
#         torch.nn.init.xavier_uniform_(self.layer2.weight)
#         torch.nn.init.xavier_uniform_(self.layer3.weight)
#         #self.model.to(device)
#         #self.epoches = epoches
#         
#     def forward(self, x):
#         #x = self.drop_out1(x)
#         x = F.relu(self.layer1(x))
#         x = F.relu(self.layer2(x))
#         x = self.layer3(x)
#        # print(x.shape)
# 
# 
#         return x
# =============================================================================


class Classfier_2layer(torch.nn.Module):
    def __init__(self,args):
        super(Classfier_2layer, self).__init__()
        self.layer1 = torch.nn.Linear(args.input_size,32)
        #self.layer2 = torch.nn.Linear(32,16)
        self.layer3 = torch.nn.Linear(32,args.output_size)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        #torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        #self.model.to(device)
        #self.epoches = epoches
        
    def forward(self, x):
        #x = self.drop_out1(x)
        x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        x = self.layer3(x)
       # print(x.shape)


        return x

class Classfier_1layer(torch.nn.Module):
    def __init__(self,args):
        super(Classfier_1layer, self).__init__()
        self.layer1 = torch.nn.Linear(args.input_size,1)
        #self.layer2 = torch.nn.Linear(32,16)
        #self.layer3 = torch.nn.Linear(32,args.output_size)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        #torch.nn.init.xavier_uniform_(self.layer2.weight)
        #torch.nn.init.xavier_uniform_(self.layer3.weight)
        #self.model.to(device)
        #self.epoches = epoches
        
    def forward(self, x):
        #x = self.drop_out1(x)
        #x = F.relu(self.layer1(x))
        #x = F.relu(self.layer2(x))
        x = self.layer1(x)
       # print(x.shape)


        return x


class Juger(torch.nn.Module):
    def __init__(self):
        super(Juger, self).__init__()
        self.layer1 = torch.nn.Conv1d(1,10,3)
        #self.act1 = torch.nn.ReLU()
        #self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
        self.layer2 = torch.nn.Conv1d(10,10,3)
        self.maxpool1=torch.nn.MaxPool1d(2)
        self.flatten = torch.nn.Flatten()
        self.layer3 = torch.nn.Linear((global_input_size-4)*5,1)
        #self.model.to(device)
        #self.epoches = epoches
        self.layer4=torch.nn.Linear(2,1)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        
    def forward(self, x,y=None):
        #x = self.drop_out1(x)
        #print(x.shape)
        x = F.relu(self.layer1(x))
        #print(x.shape)
        x = F.relu(self.layer2(x))
        #print(x.shape)
        x = self.maxpool1(x)
        #print(x.shape)
        x = self.flatten(x)
        #print(x.shape)
        
        x = self.layer3(x)
        #print(x.shape)
        #output=x
        if y is not None:
            #print(x.shape)
            #print(y.shape)
            #print(torch.cat((x,y),dim=1).shape)
            x=self.layer4(torch.cat((x,y),dim=1))

        return x


class JugerProject(torch.nn.Module):
    def __init__(self):
        super(JugerProject, self).__init__()
        self.layer1 = torch.nn.Conv1d(1,10,3)
        #self.act1 = torch.nn.ReLU()
        #self.layer2 = torch.nn.Conv1d(in_channels=20, out_channels=10, kernel_size=1)
        self.layer2 = torch.nn.Conv1d(10,10,3)
        self.maxpool1=torch.nn.MaxPool1d(2)
        self.flatten = torch.nn.Flatten()
        #self.dropout=torch.nn.Dropout(0.5)
        self.layer3 = torch.nn.Linear((global_input_size-4)*5,1)
        #self.model.to(device)
        #self.epoches = epoches
        self.layer4=torch.nn.Linear(1, (global_input_size-4)*5)
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)
        torch.nn.init.xavier_uniform_(self.layer3.weight)
        torch.nn.init.xavier_uniform_(self.layer4.weight)
        
    def forward(self, x,y=None):
        #x = self.drop_out1(x)
        #print(x.shape)
        x = F.relu(self.layer1(x))
        #print(x.shape)
        x = F.relu(self.layer2(x))
        #print(x.shape)
        x = self.maxpool1(x)
        #print(x.shape)
        x = self.flatten(x)
        #x= self.dropout(x)
        #print(x.shape)
        output = self.layer3(x)
        #print(x.shape)
        #output=x
        if y is not None:
            #print(x.shape)
            #print(y.shape)
            #print(torch.cat((x,y),dim=1).shape)
            ly=self.layer4(y)
            #print(ly.shape)
            #print(output.shape)
            #print(x.shape)
            output+=torch.sum(ly*x, dim=1, keepdim=True) 
            #print(output.shape)
            #print(output)
        return output


class Model:
    def __init__(self,args):
        if args.num_of_layer==1:
            self.model1 = Classfier_1layer(args)
        elif args.num_of_layer==2:
            self.model1 = Classfier_2layer(args)
        else:
            self.model1 = Classfier(args)
        self.model1.to(device)
        if args.juger_type==1:
            self.model2 = Juger()
        else:
            self.model2 = JugerProject()
        self.model2.to(device)
        self.epoches = args.epoch

    
    def train(self, train_set, unlab_set, elv, args, pretrain=False):
        #loss_func = nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss()
        loss_func =nn.MSELoss(reduce=False)
        #loss_func =nn.MSELoss(reduce=False, size_average=False)
        #optimizer = torch.optim.RMSprop(self.model1.parameters(),lr=0.0003)
        #optimizer = torch.optim.Adam(self.model1.parameters(),lr=args.lr, betas=(0.9, 0.99),weight_decay=0.01)
        optimizer = torch.optim.Adam(self.model1.parameters(),lr=args.lr, betas=(0.9, 0.99),weight_decay=0.05)
        optimizer_policy_gradient = torch.optim.Adam(self.model1.parameters(),lr=args.lr, betas=(0.9, 0.99),weight_decay=0.05)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max = (100 // 9) + 1)
        scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.patience, verbose=False, threshold=args.threshold, threshold_mode='rel', cooldown=args.cooldown, min_lr=0.0000001, eps=1e-08)
        scheduler_policy_gradient =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_policy_gradient, mode='min', factor=0.1, patience=args.patience, verbose=False, threshold=args.threshold, threshold_mode='rel', cooldown=args.cooldown, min_lr=0.0000001, eps=1e-08)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        #optimizer = optim.SGD(self.model1.parameters(), lr=0.001)
        if pretrain==True:
            checkpoint = torch.load('./model.pth.tar')
            for name in checkpoint ['model'].keys():
                print(name)
            # here, checkpoint is a dict with the keys you defined before
            self.model1.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['opt'])

        loss_func_judge = nn.BCEWithLogitsLoss()#nn.MSELoss(reduce=True, size_average=True)#nn.BCELoss()#nn.MSELoss(reduce=True, size_average=True)#nn.CrossEntropyLoss()
        #optimizer = torch.optim.RMSprop(self.model.parameters(),lr=0.0003)
        optimizer_judge = torch.optim.Adam(self.model2.parameters(),lr=args.lr, betas=(0.9, 0.99),weight_decay=0.05)
        scheduler_judge =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_judge, mode='min', factor=0.1, patience=args.patience, verbose=True, threshold=args.threshold, threshold_mode='rel', cooldown=args.cooldown, min_lr=0.0000001, eps=1e-08)
        for epoch in range(self.epoches):
            total_loss = 0
            total_loss1 = 0
            reward=0
           # total_loss2 = 0
            if epoch<50:
                for x in range(1000):# 每轮随机样本训练1000次
# =============================================================================
#                     train_temp = random.choices(train_set,k=5)
#                         # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
#                     #print(len(train_temp))
#                     #print(len(train_temp[0]))
#                     train_temp=np.array(train_temp)
#                     features = torch.tensor(train_temp[:,:-1], dtype=torch.float, device=device)
#                     #print(features.shape)
#                     # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
#                     label = torch.tensor(train_temp[:,-1], dtype=torch.float, device=device).unsqueeze(1)
#                     optimizer.zero_grad()
#     
#                     pred = self.model1(features) # [batch, out_dim]
#                     #print(pred.shape)
#                     #print(label.shape)
#                     loss = loss_func(pred, label)
#                     loss.backward()
#                     total_loss += loss
#                     optimizer.step()
# =============================================================================
                    train_temp = random.choices(train_set,k=20)
                        # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
                    train_temp=np.array(train_temp)
                    features = torch.tensor(train_temp[:,:-1], dtype=torch.float, device=device).unsqueeze(1)
                    #print(features.shape)
                    # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
                    label = torch.tensor(train_temp[:,-1], dtype=torch.float, device=device).unsqueeze(1)
                    optimizer.zero_grad()
    
                    pred = self.model1(features) # [batch, out_dim]
                    #print(pred.shape)
                    #print(label.shape)
                    loss = loss_func(pred, label)
                    loss.mean().backward()
                    total_loss += loss.mean().item()
                    optimizer.step()
                scheduler.step(total_loss/1000)    
                if epoch%10==0:
                    print("Classfier: in epoch {} loss {}".format(epoch, total_loss/2000))
                #print("Current Learning Rate is {}!".format(optimizer.param_groups[0]['lr']))
            else:
                if epoch==30:
                    torch.save(self.model1.state_dict(),"model.pth") # 保存参数
                    #model = model() # 代码中创建网络结构
                    params = torch.load("model.pth") # 加载参数
                    params['layer4.weight']=self.model2.layer4.weight
                    params['layer4.bias']=self.model2.layer4.bias
                    self.model2.load_state_dict(params) # 应用到网络结构中
                if epoch<120:
                    for x in range(1000):# 每轮随机样本训练1000次
                        if random.random()<=0.5:
                            train_temp = random.choices(train_set,k=20)
                            # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
                            train_temp=np.array(train_temp)
                            features = torch.tensor(train_temp[:,:-1],dtype=torch.float, device=device).unsqueeze(1)
                            #print(features.shape)
                            # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
                            label = torch.tensor(np.ones((20)), dtype=torch.float, device=device).unsqueeze(1)
                            optimizer_judge.zero_grad()
                            #pred = self.model2(features)
                            #print(features.dtype)
                            if args.juger_type==1: 
                                _y=torch.tensor(train_temp[:,-1], dtype=torch.float, device=device).unsqueeze(1)
                            else:
                                _y=torch.tensor(train_temp[:,-1], dtype=torch.float, device=device).unsqueeze(1)
                            #print(_y.dtype)
                            pred = self.model2(features,_y) # [batch, out_dim]
                            #print(pred.shape)
                            #print(pred.shape)
                            #print(label.shape)
                            #print(pred)
                            #print(label)
                            loss_judge = loss_func_judge(pred, label)
                            loss_judge.backward()
                            total_loss += loss_judge
                            optimizer_judge.step()
                        else:
                            train_temp = random.choices(unlab_set,k=20)
                            # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
                            train_temp=np.array(train_temp)
                            features = torch.tensor(train_temp, 
                                                    dtype=torch.float, device=device).unsqueeze(1)
                            #print(features.shape)
                            # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
                            label = torch.tensor(np.zeros((20)), dtype=torch.float, device=device).unsqueeze(1)
                            optimizer_judge.zero_grad()
        
                            #pred = self.model2(features)
                            #print(pred.shape)
                            #print(features.dtype)
                            if args.juger_type==1: 
                                _y=self.model1(torch.tensor(train_temp, dtype=torch.float, device=device).unsqueeze(1))
                            else:
                                _y=self.model1(torch.tensor(train_temp, dtype=torch.float, device=device).unsqueeze(1))
                                _y=torch.round(torch.sigmoid(_y))
                                #_y=_y.squeeze(1)
                                #print(_y.dtype)
                            #print(_y.dtype)
                            
                            pred = self.model2(features,_y) # [batch, out_dim]
                            #print(pred)
                            #print(label)
                            #print(pred.shape)
                            #print(label.shape)
                            loss_judge = loss_func_judge(pred, label)
                            loss_judge.backward()
                            total_loss += loss_judge
                            optimizer_judge.step()
                    scheduler_judge.step(total_loss/1000)
                    if epoch%10==0:        
                        print("Judger: in epoch {} loss {}".format(epoch, total_loss/2000))
                else:

                    for x in range(1000):# 每轮随机样本训练1000次
                        #print(x)
                        train_temp1 = random.choices(train_set,k=10)
                        train_temp1=np.array(train_temp1)
# =============================================================================
#                             # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
#                         train_temp1=np.array(train_temp1)
#                         features1 = torch.tensor(train_temp1[:,:-1],dtype=torch.float, device=device).unsqueeze(1)
#                             
#                         #print(features.shape)
#                         # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
#                         label1 = torch.tensor(np.ones((10)), dtype=torch.float, device=device).unsqueeze(1)
#                         _y1=torch.tensor(train_temp1[:,-1], dtype=torch.float, device=device).unsqueeze(1)
#                         #print(_y1.shape)
#                             #print(_y.dtype)
#                         pred1 = self.model2(features1,_y1) # [batch, out_dim]
#                         
#                         
# =============================================================================
                        train_temp2 = random.choices(unlab_set,k=10)
                        train_temp2=np.array(train_temp2)
# =============================================================================
#                         # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
#                         features2 = torch.tensor(train_temp2, 
#                                                     dtype=torch.float, device=device).unsqueeze(1)
#                         #print(features.shape)
#                         # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
#                         label2 = torch.tensor(np.zeros((10)), dtype=torch.float, device=device).unsqueeze(1)
#                         _y2=torch.round(torch.sigmoid(self.model1(features2)))
#                         #print(_y2.shape)
#                         pred2 = self.model2(features2,_y2) # [batch, out_dim]
#                         
#                         if_label=torch.cat((label1,label2))
#                         all_judge_prob=torch.cat((pred1,pred2))
#                         loss_judge=loss_func_judge(all_judge_prob,if_label)
#                         optimizer_judge.zero_grad()
#                         loss_judge.backward()
#                         total_loss1 += loss_judge
#                         optimizer_judge.step()                        
# =============================================================================
                        
                        
                        optimizer.zero_grad()
                        features3 = torch.tensor(train_temp1[:,:-1],dtype=torch.float, device=device).unsqueeze(1)
          
                        #print(features3.shape)
                        # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
                        #label3 = torch.tensor(train_temp1[:,-1], dtype=torch.float, device=device).unsqueeze(1)
                        pred3 = torch.sigmoid(self.model1(features3))
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        #print(pred3)
                       # print(label3)
                        #loss_label = loss_func(pred3, label3)
                        #print(loss_label)
                        
                        #print(pred3.shape)
                        #print(_y1.shape)
                        inverse_operation=torch.ones([10,1],dtype=torch.float, device=device)
                        _y1=torch.tensor(train_temp1[:,-1], dtype=torch.float, device=device).unsqueeze(1)
                        loss_label = abs(pred3-abs(_y1-inverse_operation))
                       #print(loss_label.shape)
                        #train_temp = random.choice(unlab_set)
                        # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
                        #features2 = torch.tensor(list(train_temp), 
                        #                        dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
                        #print(features.shape)
                        # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
                        #label2 = self.model1(features2)
                        #
# =============================================================================
#                         optimizer_judge.zero_grad()
#                         loss_label.backward()
#                         optimizer_judge.step()
# =============================================================================
                        # RNN的input要求shape为[batch, seq_len, embed_dim]，由于名字为变长，也不准备好将其填充为定长，因此batch_size取1，将取的名字放入单个元素的list中。
                        features4 = torch.tensor(train_temp2, 
                                                    dtype=torch.float, device=device).unsqueeze(1)
                        #print(features.shape)
                        # torch要求计算损失时，只提供类别的索引值，不需要one-hot表示
                        label4=torch.round(torch.sigmoid(self.model1(features4)))
                        #label4 = torch.tensor(fake_label, dtype=torch.float, device=device)
                        #pred4 = self.model1(features4)
                        #prob_unlabel_istrue_temp=torch.sigmoid(self.model2(features2,_y2)).cpu().detach().numpy().tolist()
                        #prob_unlabel_istrue=torch.tensor(prob_unlabel_istrue_temp, dtype=torch.float, device=device)
                        #print(pred4.shape)
                        #print(label4.shape)
                        #loss_unlabel_temp = prob_unlabel_istrue
                        #print(loss_unlabel_temp.shape)
                        #print(loss_unlabel_temp.shape)
                        #print(prob_unlabel_istrue.shape)
                        loss_unlabel=torch.sigmoid(self.model2(features4,label4))
                        #print(loss_unlabel.shape)
                        #print(loss_label.shape)
                        loss_total = 1-(torch.cat((loss_label,loss_unlabel),0)-0.5)
                        #loss_total = loss_label+loss_unlabel
                        #print(loss_total)
                        
        
                         # [batch, out_dim]
                        #print(pred.shape)
                        #print(pred.shape)
                        #print(label.shape)
                        #loss_judge = loss_func_judge(pred, label)
                        loss_total.mean().backward()
                        #total_loss2 += loss_total.mean()
                        reward+=loss_total.mean()
                        optimizer.step()
                        #reward+=loss_total.mean()
                    if epoch%5==0:
                        print("Judger: in epoch {} loss {}".format(epoch, total_loss1/1000.0))
                        print("Classfier: in epoch {} reward {}".format(epoch, reward/1000))
                    if epoch%30==0:
                        self.evaluate(elv)
                        #print(loss_total)
                    scheduler_policy_gradient.step(reward/1000)
                    scheduler_judge.step(total_loss1/1000)
    def evaluate(self, test_set):
        with torch.no_grad(): # 评估时不进行梯度计算
            features = torch.tensor(test_set[:,:-1], dtype=torch.float, device=device).unsqueeze(1)
            final_prediction = self.model1(features)
            final_pred_np = final_prediction.cpu().detach().numpy()
            temp=final_pred_np.squeeze()
            #temp[np.where(temp>=0.5)]=1
            #temp[np.where(temp<0.5)]=0
            print('Evaluating: test accuracy is {}'.format(roc_auc_score(test_set[:,-1],temp )))

            #print('Evaluating: test accuracy is {}%'.format(correct*100/np.size(test_set,0)))
            return np.corrcoef(final_pred_np.squeeze(), test_set[:,-1])[0]
        
    def predict(self, name,word):
        #p = name2vec(name.lower())
        name_tensor = torch.tensor(name, dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = self.model1(name_tensor)
            out = torch.argmax(out).item()
            sexy = 'gamble' if out == 0 else 'no-gamble'
            print('{} is {}'.format(word, sexy))


 
def main(args):



    model5=Model(args)
    training_data1=np.concatenate((X1,np.expand_dims(Y1, axis=1)),axis=1)
    training_data2=np.concatenate((X2,np.expand_dims(Y2, axis=1)),axis=1)
    model5.train(training_data1,X3,training_data2,args)
    #grid_search_40_2000.append(model5.evaluate(training_date2))
        
    #print(model5.model1(torch.tensor(X3[1], dtype=torch.float, device=device)))
    torch.save(model5, './adv_learning_model.pkl')
    predict_result=torch.zeros(len(X3),1,dtype=torch.float32)
    with torch.no_grad():
        for i in range(int(len(X3)/100)+1):
            predict_result[i*100:min((i+1)*100,len(X3))]=model5.model1(torch.tensor(X3[i*100:min((i+1)*100,len(X3)),:], dtype=torch.float, device=device).unsqueeze(1))
    return predict_result
    # =============================================================================
    # model5.train(training_data1,pretrain=True)
    # #grid_search_40_2000.append(model5.evaluate(training_date2))
    #     
    # print(model5.model(torch.tensor(X3[1], dtype=torch.float, device=device).unsqueeze(0).unsqueeze(0)))
    # model5.evaluate(training_data1)
    # =============================================================================
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
#from sklearn.metrics import score
import pickle
if __name__ == "__main__":
    X1, Y1,X2,Y2,X3,Z1 = get_data(0.01) 
    

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=0.01)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batchsize", type=int, default=1)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--input_size", type=int, default=768)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--threshold", type=int, default=0.00001)
    parser.add_argument("--cooldown", type=int, default=0)
    parser.add_argument("--num_of_layer", type=int, default=3)
    parser.add_argument("--juger_type", type=int, default=2)
    args = parser.parse_args()

      

    args.patience=3
    #args.threshold=0.0001
    args.cooldown=2
    args.num_of_layer=3
    #pred_adv=main(args)
    model5=Model(args)
    training_data1=np.concatenate((X1,np.expand_dims(Y1, axis=1)),axis=1)
    training_data2=np.concatenate((X2,np.expand_dims(Y2, axis=1)),axis=1)
    model5.train(training_data1,X3,training_data2,args)
    #grid_search_40_2000.append(model5.evaluate(training_date2))
        
    #print(model5.model1(torch.tensor(X3[1], dtype=torch.float, device=device)))
    torch.save(model5, './adv_learning_model.pkl')
    with open('../nlp/gamble_new_candidate_words.pickle', "rb") as input_file:
         word_index = pickle.load(input_file)
    idxs_index=[]
    idxs_words=[]
    for key in word_index:
        if key in Z1:
            idxs_index.append(np.where(Z1==key)[0][0])
            idxs_words.append(key)
    current_input=X3[idxs_index,:]
    predict_result=torch.zeros(len(current_input),1,dtype=torch.float32)
    with torch.no_grad():
        for i in range(int(len(current_input)/100)+1):
            predict_result[i*100:min((i+1)*100,len(current_input))]=model5.model1(torch.tensor(current_input[i*100:min((i+1)*100,len(current_input)),:], dtype=torch.float, device=device).unsqueeze(1))

    pred_adv_np=predict_result.cpu().detach().numpy()
    sort_of_adv=sorted(range(len(pred_adv_np)), key=lambda k: -pred_adv_np[k])
    output_adv=np.array(idxs_words)[sort_of_adv[:500]]
    
    with open('../nlp/wahaha_words.pickle', "rb") as input_file:
         word_index = pickle.load(input_file)
    idxs_index=[]
    idxs_words=[]
    for key in word_index:
        if key in Z1:
            idxs_index.append(np.where(Z1==key)[0][0])
            idxs_words.append(key)
    current_input=X3[idxs_index,:]
    predict_result2=torch.zeros(len(current_input),1,dtype=torch.float32)
    with torch.no_grad():
        for i in range(int(len(current_input)/100)+1):
            predict_result2[i*100:min((i+1)*100,len(current_input))]=model5.model1(torch.tensor(current_input[i*100:min((i+1)*100,len(current_input)),:], dtype=torch.float, device=device).unsqueeze(1))

    pred_adv_np2=predict_result2.cpu().detach().numpy()
    sort_of_adv2=sorted(range(len(pred_adv_np2)), key=lambda k: -pred_adv_np2[k])
    output_adv2=np.array(idxs_words)[sort_of_adv2[:500]]
# =============================================================================
    import re
    def is_good(w):
         if re.findall(u'[\u4e00-\u9fa5]', w) \
             and len(w) >= 2\
             and not re.findall(u'[较很越增]|[多少大小长短高低好差]', w)\
             and not (u'代上级' in w or u'唔知' in w or u'唔该' in w or u'曾经' in w  or u'施主' in w or u'为何' in w or u'风流' in w or u'招聘' in w or u'夏令营' in w or u'继续' in w or u'跆拳道' in w or u'西红柿' in w or u'满群' in w or u'汽油' in w or u'处理' in w or u'霸王餐' in w or u'帅哥美女' in w or u'兄弟姐妹' in w or u'党费' in w or u'资料费' in w or u'电费' in w or u'生活费' in w or u'车费' in w or u'气费' in w or u'报名费' in w or u'出现' in w or u'有求' in w or u'情谊' in w or u'永不' in w  or u'坚决' in w)\
             and not w[-1] in u'们啦我你他投图机抢的送班购了吗群轮店货日啊好的个是国春爱哟哦'\
             and not w[:1] in [u'求',u'情',u'杨',u'心',u'昨',u'那',u'赚',u'给',u'这',u'收',u'今',u'送',u'的',u'祝',u'每',u'不',u'有',u'你',u'我',u'他',u'她',u'它']\
             and not w[:3] in [u'朋友圈']\
             and not w[-2:] in [u'开始',u'返现',u'新人',u'求助',u'完成',u'烦躁',u'守候',u'基金',u'志明',u'春娇',u'殿明',u'兑现',u'不起']\
             and not w[2:] in [u'发起',u'圆满',u'推荐',u'慈善',u'美好',u'冻结']\
             and not w[-3:] in [u'倒计时',u'玻尿酸',u'睫毛膏',u'精华液']:
             return True
         else:
             return False
#         
    import AhoCorasickTree as ACT    
    def build_ac_tree(pattern_word_list):
         
         ac = ACT.ACTree()
         ac.build(pattern_word_list)
         
         return ac
#     
    fields=['key']    
    df_city = pd.read_csv('./data/citylist.csv',encoding='utf-8', usecols=fields)
    city_list=np.squeeze(np.array(df_city.values.tolist())).tolist()
    AC_Tree_City=build_ac_tree(city_list)
    def is_not_city(w):
             match_res = AC_Tree_City.match(w)
             words_set=set(match_res)
             if not words_set:
                 return True
             else:
                 return False
    import codecs
    name_list = codecs.open('./data/Chinese_Names_120W.txt', 'r', 'utf-8').read().split()
    chengyu_list = codecs.open('./data/ChengYu_5W.txt', 'r', 'utf-8').read().split()
    relationship_list = codecs.open('./data/Chinese_Relationship.txt', 'r', 'utf-8').read().split()
 
 
    AC_Tree_name_chengyu=build_ac_tree(name_list+chengyu_list+relationship_list)
    def is_not_name_or_chengyu(w):
         match_res = AC_Tree_name_chengyu.match(w)
         words_set=set(match_res)
         if not words_set:
             return True
         else:
             return False
# =============================================================================
    output_version='1120'
    word_output_adv_clear=[i for i in output_adv if is_good(i) and is_not_city(i) and is_not_name_or_chengyu(i)]
    pd.DataFrame(np.array(word_output_adv_clear).reshape((len(word_output_adv_clear),1))).to_csv('york_gamble_newwords'+output_version+'_adv.csv',encoding='utf-8-sig')
    
        
    clf = svm.SVC(gamma='scale',probability=True, kernel = 'linear')
    clf.fit(X1, Y1)
    y_pred_svm=clf.predict(current_input)
    sort_of_svm=sorted(range(len(y_pred_svm)), key=lambda k: -y_pred_svm[k])
    output_svm=np.array(idxs_words)[sort_of_svm[0:500]]
    word_output_svm_clear=[i for i in output_svm if is_good(i) and is_not_city(i) and is_not_name_or_chengyu(i)]
    pd.DataFrame(np.array(word_output_svm_clear).reshape((len(word_output_svm_clear),1))).to_csv('york_gamble_newwords'+output_version+'_svm.csv',encoding='utf-8-sig')
    
    
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(global_input_size, global_input_size), random_state=1)
    clf.fit(X1, Y1)                     
    y_pred_mlp=clf.predict(current_input)
    sort_of_mlp=sorted(range(len(y_pred_mlp)), key=lambda k: -y_pred_mlp[k])
    output_mlp=np.array(idxs_words)[sort_of_mlp[0:500]]
    word_output_mlp_clear=[i for i in output_mlp if is_good(i) and is_not_city(i) and is_not_name_or_chengyu(i)]
    pd.DataFrame(np.array(word_output_mlp_clear).reshape((len(word_output_mlp_clear),1))).to_csv('york_gamble_newwords'+output_version+'_mlp.csv',encoding='utf-8-sig')
      