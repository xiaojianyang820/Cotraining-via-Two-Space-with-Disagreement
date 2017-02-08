# -*- coding: utf-8 -*-
"""
Created on Wed Feb 08 14:52:46 2017

@author: Administrator
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
import random
from sklearn.metrics import roc_curve,auc
from sklearn.preprocessing import StandardScaler

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['FangSong'] # 解决中文显示方块的问题
mpl.rcParams['axes.unicode_minus'] = False # 解决保存图像是负号'-'显示为方块的问题

class FANS(object):
    
    def __init__(self,train_data,train_labels,test_data,bw_method='silverman',
                 iteration=4,CS=np.arange(0.1,10,0.1),threshold=0.005):
        '''
            请同时输入有标签组train_data,train_labels以及无标签组test_data,test_labels
            
            其他可选参数包括：
            
                高斯核密度估计选择窗宽所依据的准则 bw_method：'scott','silverman'或者某个自定义的可调用函数，默认是'silverman'
                
                切分次数iteration：对有标签组做多少次切分来进行投票表决，默认值是4
                
                L1正则交叉验证的可取值区间CS：默认值是np.arange（0.1,10,0.2）
                
                稳定性阈值threshold：默认值为0.005
        '''
        self.train_data = np.array(train_data)
        self.train_labels = np.array(train_labels)
        self.test_data = np.array(test_data)
        self.train_num = self.train_data.shape[0]
        self.test_num = self.test_data.shape[0]
        self.feature_num = self.train_data.shape[1]
        assert self.train_data.shape[1] == self.test_data.shape[1]
        self.bw_method = bw_method
        self.iteration = iteration
        self.CS = CS
        self.threshold = threshold
        
    def Main(self):
        '''
            FANS算法的主流程
        '''
        voted_pro_predict = []
        for t in range(self.iteration):
            print '.'
            KFOLD = cross_validation.KFold(self.train_num,n_folds=2,shuffle=True)
            for KDE_part,PLR_part in KFOLD:
                X_KDE,Y_KDE = self.train_data[KDE_part],self.train_labels[KDE_part]
                X_PLR,Y_PLR = self.train_data[PLR_part],self.train_labels[PLR_part]
                transformed_X_PLR = self.CON_TRANER(X_KDE,Y_KDE,X_PLR)
                transformed_X_test = self.CON_TRANER(X_KDE,Y_KDE,self.test_data)
                PLR_CLF = LogisticRegressionCV(penalty='l1',Cs=self.CS,cv=3,solver='liblinear')
                PLR_CLF.fit(transformed_X_PLR,Y_PLR)
                PLR_pro_predict = PLR_CLF.predict_proba(transformed_X_test)[:,1]
                voted_pro_predict.append(PLR_pro_predict)
        voted_pro_predict = np.array(voted_pro_predict,dtype=float)
        final_pro_predict = np.sum(voted_pro_predict,axis=0)/voted_pro_predict.shape[0]
        return final_pro_predict
        
    def CON_TRANER(self,X_KDE,Y_KDE,X_PLR):
        transformed_data = []
        X_PLR = X_PLR.T
        KDE_Generator = self.CON_KDE_Generator(X_KDE,Y_KDE)
        modify = lambda pro:np.max([self.threshold,pro])
        for item in X_PLR:
            item_set = set(item)
            pro_dict = {}
            KDE_0,KDE_1 = KDE_Generator.next()
            for i in item_set:
                pro_0,pro_1 = KDE_0(i),KDE_1(i)
                pro_0 = np.array(map(modify,pro_0))
                pro_1 = np.array(map(modify,pro_1))
                pro_dict[i] = np.log(pro_1/pro_0)
            map_pro = lambda value:pro_dict[value]
            item = np.array(map(map_pro,item))[:,0]
            transformed_data.append(item)
        return np.array(transformed_data).T
        
    def CON_KDE_Generator(self,Xs,Ys):
        '''
            协程：核密度估计生成器
        '''
        for item in range(self.feature_num):
            s_0 = Xs[Ys==0][:,item]
            s_1 = Xs[Ys==1][:,item]
            KDE_0 = self.KDE(s_0)
            KDE_1 = self.KDE(s_1)
            yield KDE_0,KDE_1
    
    def KDE(self,X):
        '''
            核密度估计
        '''
        try:
            return stats.gaussian_kde(X,bw_method=self.bw_method)
        except:
            X += np.random.randn(X.shape[0])*0.1
            return stats.gaussian_kde(X,bw_method=self.bw_method)
            
def PLR(train_data,train_label,test_data,CS=np.arange(0.1,1,0.05)):
    whole_data = np.vstack((train_data,test_data))
    S = StandardScaler(with_mean=True,with_std=True)
    S.fit(whole_data)
    train_data = S.transform(train_data)
    test_data = S.transform(test_data)
    
    CLF = LogisticRegressionCV(penalty='l1',Cs=CS,cv=3,solver='liblinear')
    CLF.fit(train_data,train_label)
    pro_predict = CLF.predict_proba(test_data)[:,1]
    return pro_predict
    
class Cotraining(object):
    def __init__(self,X_train,Y_train,X_test,iteration=10,FANS_CS=np.arange(0.1,10,0.1),
                 PLR_CS=np.arange(0.1,0.5,0.05),times=1.0):
        """
            初始化各参数意义：
                1.X_train: 有标签组的特征
                2.Y_train: 有标签组的标签
                3.X_test : 无标签组的特征
                4.iteration: 总迭代次数
                5.FANS_CS: FANS算法正则系数交叉验证候选集
                6.PLR_CS: PLR算法正则系数交叉验证候选集
                7.times: 扩张系数
            Cotraining对象中其他属性：
                1.FANS_train_set_X: FANS算法的训练集特征
                2.FANS_train_set_Y: FANS算法的训练集标签
                3.PLR_train_set_X:  PLR算法的训练集特征
                4.PLR_train_set_Y:  PLR算法的训练集标签
                5.state_dict     :  无标签组各样本点的状况，详见making_dict函数的说明
        """
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.FANS_train_set_X = self.X_train
        self.FANS_train_set_Y = self.Y_train
        self.PLR_train_set_X = self.X_train
        self.PLR_train_set_Y = self.Y_train
        self.state_dict = self.making_dict()
        self.iteration = iteration
        self.FANS_high_confidence = []
        self.PLR_high_confidence = []
        self.FANS_CS = FANS_CS
        self.PLR_CS = PLR_CS
        self.times = times
        self.Evaluation = {}
        
    def start(self,istest=0,Y_test=None):
        """
            双空间分歧协同模型的主进程。
            如果istest参数设置为1，则需要同时传入Y_test参数，以对模型的效果进行检验。
        """
        zhibiao = ['accu','auc','nums','high_nums','FANS_nums','PLR_nums']
        for item in zhibiao:
            self.Evaluation[item] = []
        for item in range(self.iteration):
            print u'第%d次循环'%item
            fans = FANS(self.FANS_train_set_X,self.FANS_train_set_Y,self.X_test,CS=self.FANS_CS)
            FANS_pro_predict = fans.Main()
            self.FANS_select(FANS_pro_predict)
            if istest:
                auc,accu = self.evaluate(Y_test,FANS_pro_predict,"FANS")
                print u'FANS算法的AUC值: %.4f'%auc
                print u'当前高置信组的正确率: %.4f'%accu
            plr_pro_predict = PLR(self.PLR_train_set_X,self.PLR_train_set_Y,self.X_test,CS=self.PLR_CS)
            self.PLR_select(plr_pro_predict)
            if istest:
                auc,accu = self.evaluate(Y_test,plr_pro_predict,"PLR")
                print u'PLR算法的AUC值: %.4f'%auc
                print u'当前高置信组的正确率: %.4f'%accu
        prodict = np.vstack((FANS_pro_predict,plr_pro_predict))
        prodict = np.mean(prodict,axis=0)
        for i,item in enumerate(self.state_dict[:,0]):
            if item == 1:
                label = self.state_dict[i,1]
                if label == 1:
                    label=1.1
                else:
                    if label == 0:
                        label=-0.1
                prodict[i] = label
        return prodict
            
        
    def making_dict(self):
        '''
            无标签组U的状态字典state_dict
            第一位置是是否进入高置信组
            第二位置是伪标签
            第三位置是在历次循环中算法给标签
            第四位置是第一次进入高置信组时的置信度
        '''
        state_dict = []
        for item in range(self.X_test.shape[0]):
            state_dict.append([0,-1,[],-1])
        return np.array(state_dict,dtype=object)
        
    def FANS_select(self,FANS_pro_predict):
        certain_index = self.selected_2(FANS_pro_predict,self.FANS_train_set_X.shape[0])
        for i,item in enumerate(certain_index):
            if item == True:
                self.state_dict[i][2].append(FANS_pro_predict[i])
                if self.state_dict[i][0] == 1:
                    continue
                else:
                    self.state_dict[i][0] = 1
                    self.state_dict[i][1] = self.pro2int([FANS_pro_predict[i]])[0]
                    self.state_dict[i][3] = FANS_pro_predict[i]
        sub_certain_index = []
        for i,item in enumerate(certain_index):
            if (item == True) and (i not in self.PLR_high_confidence):
                sub_certain_index.append(True)
                self.PLR_high_confidence.append(i)
            else:
                sub_certain_index.append(False)
        certain_index = np.array(sub_certain_index)
        psudo_X = self.X_test[certain_index]
        psudo_Y = self.pro2int(FANS_pro_predict[certain_index])
        self.PLR_train_set_X = np.vstack((self.PLR_train_set_X,psudo_X))
        self.PLR_train_set_Y = np.hstack((self.PLR_train_set_Y,psudo_Y))
            
    def PLR_select(self,PLR_pro_predict):
        certain_index = self.selected_2(PLR_pro_predict,self.PLR_train_set_X.shape[0])
        for i,item in enumerate(certain_index):
            if item == True:
                self.state_dict[i][2].append(PLR_pro_predict[i])
                if self.state_dict[i][0] == 1:
                    continue
                else:
                    self.state_dict[i][0] = 1
                    self.state_dict[i][1] = self.pro2int([PLR_pro_predict[i]])[0]
                    self.state_dict[i][3] = PLR_pro_predict[i]
        sub_certain_index = []
        for i,item in enumerate(certain_index):
            if (item == True) and (i not in self.FANS_high_confidence): 
                sub_certain_index.append(True)
                self.FANS_high_confidence.append(i)
            else:
                sub_certain_index.append(False)
        certain_index = np.array(sub_certain_index)
        psudo_X = self.X_test[certain_index]
        psudo_Y = self.pro2int(PLR_pro_predict[certain_index])
        self.FANS_train_set_X = np.vstack((self.FANS_train_set_X,psudo_X))
        self.FANS_train_set_Y = np.hstack((self.FANS_train_set_Y,psudo_Y))
        
    def selected_2(self,pro_predict,num):
        pro_predict = np.abs(pro_predict - 0.5)
        arged_index = np.argsort(pro_predict)[-int(num*self.times):]
        certain_index = self.X_test.shape[0]*[False]
        for item in arged_index:
            certain_index[item] = True
        return np.array(certain_index)
        
    def pro2int(self,pro):
        data = []
        for item in pro:
            if item > 0.5:
                data.append(1)
            else:
                data.append(0)
        return np.array(data)
        
    def evaluate(self,Y_test,pro_predict,method):
        for i,item in enumerate(self.state_dict[:,0]):
            if item == 1:
                label = self.state_dict[i,3]
                pro_predict[i] = label
        int_pro_predict = self.pro2int(pro_predict)
        print method+'准确度：%.4f'%np.mean(int_pro_predict == Y_test)
        self.Evaluation['accu'].append(np.mean(int_pro_predict == Y_test))
        False_positive_rate,True_positive_rate,thresholds = roc_curve(Y_test,pro_predict)
        roc_auc = auc(False_positive_rate,True_positive_rate)
        accu = 0.0
        total = 0.0
        for i,item in enumerate(Y_test):
            if self.state_dict[i][0] == 1:
                total += 1
                if item == self.state_dict[i][1]:
                    accu += 1
        print u'高置信组的规模: %d'%total
        self.Evaluation['auc'].append(roc_auc)
        self.Evaluation['nums'].append(total)
        self.Evaluation['high_nums'].append(accu/total)
        self.Evaluation['FANS_nums'].append(len(self.FANS_high_confidence))
        self.Evaluation['PLR_nums'].append(len(self.PLR_high_confidence))
        return roc_auc,accu/total
