#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import random
import re
import math
import numpy as np


Container_Num = 100
EN_Num = 30
#resource_total_EN =np.array([[32, np.random.randint(16,65), np.random.randint(100,400)]for i in range(10)])
#resource_Container =np.array([[np.random.randint(1,3),np.random.randint(1,3),np.random.randint(2,15)]for i in range(Container_Num)])
#data_trans_Container = np.random.randint(100,500,Container_Num)
#Band = np.triu(np.random.randint(100,300,100).reshape(10, 10))
#Band += Band.T - np.diag(Band.diagonal())
#Band = Band - np.diag(np.diag(Band))


resource_used_EN = np.zeros([EN_Num,3], dtype=int)
resource_remaining_EN = np.zeros([EN_Num,3], dtype=int)
resource_utilization_EN = np.zeros([EN_Num,3], dtype=int)
load_EN = np.zeros(EN_Num)
load_differentiation_EN = np.zeros([EN_Num,EN_Num])
#X = np.zeros([Container_Num,10], dtype=int)
#for i in range(Container_Num):
    #X[i][np.random.randint(0,10)]=1
resource_total_EN = np.loadtxt('D:\ACS16\Resource_total_EN.txt', dtype='int', delimiter=',')
resource_Container = np.loadtxt('D:\ACS16\Resource_Container.txt', dtype='int', delimiter=',')
data_trans_Container = np.loadtxt('D:\ACS16\data_trans_Container.txt', dtype='int', delimiter=',')
Band = np.loadtxt('D:\ACS16\Band.txt', dtype='int', delimiter=',')
X = np.loadtxt('D:\ACS16\X.txt', dtype='int', delimiter=',')
print('边缘节点资源总量：\n', resource_total_EN)
print('容器资源需求：\n', resource_Container)
print('容器数据传输总量：\n', data_trans_Container)
print('边缘节点间带宽：\n', Band)
print('部署决策矩阵：\n', X)
Y = np.copy(X)
for i in range(Container_Num):
    for j in range(EN_Num):
        if X[i][j] == 1:
            resource_used_EN[j][0] += resource_Container[i][0]
            resource_used_EN[j][1] += resource_Container[i][1]
            resource_used_EN[j][2] += resource_Container[i][2]
            break
        else:
            continue
resource_remaining_EN = resource_total_EN-resource_used_EN
resource_utilization_EN = np.around(resource_used_EN/resource_total_EN, decimals=2)
for i in range(EN_Num):
    load_EN[i] = (resource_utilization_EN[i][0]+resource_utilization_EN[i][1]+resource_utilization_EN[i][2])/3
for i in range(EN_Num):
    for j in range(EN_Num):
        load_differentiation_EN[i][j] = load_EN[i]/load_EN[j]
load_differentiation_EN = np.around(load_differentiation_EN, decimals=2)
print('边缘节点已用资源：\n', resource_used_EN)
print('边缘节点剩余资源：\n', resource_remaining_EN)
print('边缘节点资源利用率：\n', resource_utilization_EN)
print('边缘节点负载：\n', load_EN)
print('边缘节点间负载差异化：\n', load_differentiation_EN)


# 负载均衡检测
def Overload_Detection():
    temp = []
    for i in range(EN_Num):
        for j in range(EN_Num):
            if load_differentiation_EN[i][j] >= 2.2:
                temp.append(i)
                break
            else:
                continue
    for i in range(EN_Num):
        if resource_utilization_EN[i][0] >= 0.6 or resource_utilization_EN[i][1] >= 0.9 or resource_utilization_EN[i][2] >= 0.9:
            temp.append(i)
    return temp
overload_EN = list(set(Overload_Detection()))
print('过载边缘节点：\n', overload_EN)


# 获取边缘节点上的容器集合
def getassignedContainerList(i):
    ContainerList_total = []
    for j in range(Container_Num):
        if X[j][i] == 1:
            ContainerList_total.append(j)
    return ContainerList_total


# 获取待迁移容器集合
def Container_Selection():
    migration_probability_total =[]
    container_probability_total = []
    mig_Container = []
    mig_Container_total = []
    for i in list(set(Overload_Detection())):  # for i in overload_EN
        sum_resource_utilization_EN = resource_utilization_EN[i][0] + resource_utilization_EN[i][1] + resource_utilization_EN[i][2]
        ContainerList_total = getassignedContainerList(i)
        for j in ContainerList_total:
            mig_time = (resource_Container[j][0] / resource_used_EN[i][0]+resource_Container[j][1] / resource_used_EN[i][1]) / 2
            mig_number = 0
            #mig_quantity = random.random()
            mig_quantity = data_trans_Container[j] / 1000
            for k in range(3):
                mig_number += (resource_utilization_EN[i][k] / sum_resource_utilization_EN)/math.sqrt(math.pow((resource_utilization_EN[i][k]-resource_Container[j][k]/resource_total_EN[i][k]),2))
            migration_probability = (mig_time+mig_number-mig_quantity)/3
            migration_probability_total.append(migration_probability)
        container_probability = dict(zip(ContainerList_total, migration_probability_total))
        container_probability = dict(sorted(container_probability.items(),key=lambda x:x[1],reverse=True))
        # print(container_probability)      #按迁移概率降序排序之后的字典{容器:迁移概率}
        container_probability_total.append(container_probability)
    # print(container_probability_total)    #全部过载节点上的容器与迁移概率
    EN_Container = dict(zip(list(set(Overload_Detection())), container_probability_total))
    print(EN_Container)           # {过载边缘节点:{部署容器:迁移概率}}
    for i in EN_Container.keys():
        Container_assigned = EN_Container[i].keys()
        for j in Container_assigned:
            mig_Container.append(j)
            resource_used_EN[i][0] -= resource_Container[j][0]
            resource_used_EN[i][1] -= resource_Container[j][1]
            resource_used_EN[i][2] -= resource_Container[j][2]
            Y[j][i] = 0
            if resource_used_EN[i][0]/resource_total_EN[i][0] < 0.6 and resource_used_EN[i][1]/resource_total_EN[i][1] < 0.9 and resource_used_EN[i][2]/resource_total_EN[i][2] < 0.9:
                break
            else:
                continue
        resource_remaining_EN[i] = resource_total_EN[i] - resource_used_EN[i]
        resource_utilization_EN[i] = np.around(resource_used_EN[i] / resource_total_EN[i], decimals=2)
    return mig_Container
migrated_ContainerList = Container_Selection()
migrated_ContainerList_Num = len(migrated_ContainerList)
print("待迁移容器集合：\n",migrated_ContainerList)
print("迁移后节点资源利用率：\n",resource_utilization_EN)
print("迁移后节点剩余资源：\n",resource_remaining_EN)


# 获取待迁移容器所在边缘节点
def getENbeforemigrating(i,X):
    EN_beforemigrating = 0
    for j in range(EN_Num):
        if X[i][j]==1:
            EN_beforemigrating =j
            break
    return EN_beforemigrating


# 获取容器的有效边缘节点集合
def getavailableENSet(l):
    available_EN = []
    jj = getENbeforemigrating(l,X)
    for j in range(EN_Num):
        if (resource_used_EN[j][0] + resource_Container[l][0])/resource_total_EN[j][0] < 0.6 and (resource_used_EN[j][1] + resource_Container[l][1])/resource_total_EN[j][1] < 0.9 and (resource_used_EN[j][2] - resource_Container[l][2])/resource_total_EN[j][2] < 0.9 and j != jj:
            available_EN.append(j)
    return available_EN


# 初始化第一代染色体
def Initialize_first_generation_chromosomes(chromosomeNum):
    Y_allChromo = []
    global Y
    copy_Y = np.copy(Y)
    for chromo in range(chromosomeNum):
        for i in migrated_ContainerList:
            temp = random.randint(0, 4)
            Y[i][temp] = 1
        Y_allChromo.append(Y)
        Y = np.copy(copy_Y)
    return Y_allChromo


# 计算适应度函数值
def calAdaptability_chromosome(Z_allChromo):
    Adaptability_allChromo = []
    for chromo in range(len(Z_allChromo)):
        punish = 0
        resource_used_Matrix = np.zeros([EN_Num, 3], dtype=int)
        resource_remaining_Matrix = np.zeros([EN_Num, 3], dtype=int)
        resource_utilization_Matrix = np.zeros([EN_Num, 3], dtype=int)
        resource_uti_total_0 = resource_uti_total_1 = resource_uti_total_2 = 0
        resource_bal_total_0 = resource_bal_total_1 = resource_bal_total_2 = 0
        resource_remaining_total = 0
        migration_cost_total = 0
        for i in range(Container_Num):
            for j in range(EN_Num):
                if Z_allChromo[chromo][i][j] == 1:
                    resource_used_Matrix[j][0] += resource_Container[i][0]
                    resource_used_Matrix[j][1] += resource_Container[i][1]
                    resource_used_Matrix[j][2] += resource_Container[i][2]
                    break
                else:
                    continue
        resource_remaining_Matrix = resource_total_EN - resource_used_Matrix
        resource_utilization_Matrix = np.around(resource_used_Matrix / resource_total_EN, decimals=2)
        for i in range(EN_Num):
            if resource_utilization_EN[i][0] >= 0.6 or resource_utilization_EN[i][1] >= 0.9 or resource_utilization_EN[i][2] >= 0.9:
                punish = 1
                break
        for i in range(EN_Num):
            resource_uti_total_0 += resource_utilization_Matrix[i][0]
            resource_uti_total_1 += resource_utilization_Matrix[i][1]
            resource_uti_total_2 += resource_utilization_Matrix[i][2]
        resource_average_0 = resource_uti_total_0 / EN_Num
        resource_average_1 = resource_uti_total_1 / EN_Num
        resource_average_2 = resource_uti_total_2 / EN_Num
        for i in range(EN_Num):
            resource_bal_total_0 += (resource_utilization_Matrix[i][0] - resource_average_0) ** 2 / EN_Num
            resource_bal_total_1 += (resource_utilization_Matrix[i][1] - resource_average_1) ** 2 / EN_Num
            resource_bal_total_2 += (resource_utilization_Matrix[i][2] - resource_average_2) ** 2 / EN_Num
        resource_balancing_total = resource_bal_total_0 + resource_bal_total_1 + resource_bal_total_2
        for i in range(EN_Num):
            resource_remaining_total +=(abs(resource_utilization_Matrix[i][1]-resource_utilization_Matrix[i][0])+abs(resource_utilization_Matrix[i][2]-resource_utilization_Matrix[i][0])+abs(resource_utilization_Matrix[i][2]-resource_utilization_Matrix[i][1]))
        if (Z_allChromo[chromo] == X).all():
            migration_cost_total = 0
        else:
            for i in migrated_ContainerList:
                EN_before = getENbeforemigrating(i,X)
                EN_after = getENbeforemigrating(i,Z_allChromo[chromo])
                if EN_before==EN_after:
                    punish = 1
                    migration_cost_total += 0
                else:
                    migration_cost_total += data_trans_Container[i]/(1024*Band[EN_before][EN_after]/1000)
        Optimization_Target = (resource_balancing_total+resource_remaining_total+migration_cost_total)/3
        Adaptability = 20-Optimization_Target-punish*10
        Adaptability_allChromo.append(Adaptability)
    return Adaptability_allChromo

# 计算自然选择概率
def calSelectionProbability(Adaptability_all,chr_Num):
    sumAdaptability = 0
    selectionProbability = []
    for i in range(chr_Num):
        sumAdaptability += Adaptability_all[i]
    for i in range(chr_Num):
        selectionProbability.append(Adaptability_all[i]/sumAdaptability)
    return selectionProbability


def wheel(selectionProbability_allChromo):
    wheel_value = random.random()
    probability_total = 0
    for i in range(len(selectionProbability_allChromo)):
        probability_total += selectionProbability_allChromo[i]
        if probability_total >= wheel_value:
            temp = i
            break
        else:
            continue
    return temp


# 交叉
def cross(Y_allChromo, selectionProbability_allChromo):
    global Y
    copy_Y = np.copy(Y)
    newY_allChromo = []
    for index in range(16):
        Y_dad_List = []
        Y_mom_List = []
        Y_dad = Y_allChromo[wheel(selectionProbability_allChromo)]
        Y_mom = Y_allChromo[wheel(selectionProbability_allChromo)]
        for i in migrated_ContainerList:
            for j in range(EN_Num):
                if Y_dad[i][j] ==1:
                    Y_dad_List.append(j)
                    break
        for i in migrated_ContainerList:
            for j in range(EN_Num):
                if Y_mom[i][j] ==1:
                    Y_mom_List.append(j)
                    break
        crossIndex = random.randint(1, migrated_ContainerList_Num - 1)
        Y_child_List = Y_dad_List[0:crossIndex]+Y_mom_List[crossIndex:]
        index = 0
        for i in migrated_ContainerList:
            Y[i][Y_child_List[index]] =1
            index += 1
        newY_allChromo.append(Y)
        Y = np.copy(copy_Y)
    return newY_allChromo


# 复制
def copy(Y_allChromo, Adaptability_allChromo, newY_allChromo):
    Adaptability_allChromo = np.array(Adaptability_allChromo)
    max_min_List = np.argsort(-Adaptability_allChromo)
    temp = 0
    for i in range(4):
        Y_oneChromo = Y_allChromo[max_min_List[temp]]
        newY_allChromo.append(Y_oneChromo)
        temp += 1
    return newY_allChromo


# 变异
def mutation(newY_allChromo):
    for i in range(4):
        chromosomeIndex = random.randint(0, 15)
        ContainerIndex = random.choice(migrated_ContainerList)
        temp = getENbeforemigrating(ContainerIndex,newY_allChromo[chromosomeIndex])
        newY_allChromo[chromosomeIndex][ContainerIndex][temp] = 0
        ENIndex = random.randint(0,4)
        newY_allChromo[chromosomeIndex][ContainerIndex][ENIndex] = 1
    return newY_allChromo


def createGeneration(Y_allChromo,Adaptability_allChromo,selectionProbability_allChromo):
    newY_allChromo = cross(Y_allChromo,selectionProbability_allChromo)
    newY_allChromo = mutation(newY_allChromo)     # 变异
    newY_allChromo = copy(Y_allChromo, Adaptability_allChromo,newY_allChromo)      # 复制
    return newY_allChromo


def getmaxAdaptability(Adaptability_allChromo):
    max_Optimization_target = Adaptability_allChromo[0]
    for i in range(len(Adaptability_allChromo)):
        if Adaptability_allChromo[i] > max_Optimization_target:
            max_Optimization_target = Adaptability_allChromo[i]
    return 20-max_Optimization_target


# 容器部署
def Container_Placement_Based_On_GA():
    iteratorNum =300  # 迭代次数
    chromosomeNum = 20      # 染色体个数
    cp = 0.2          # 复制概率
    p_c = 0.8         # 交叉概率
    p_m = 0.2          # 变异概率
    Y_allChromo = []  # 一次迭代 -> 10个决策数组
    Adaptability_allChromo = []  # 一次迭代 -> 10个适应度目标值
    selectionProbability_allChromo = []
    Y_allChromo = Initialize_first_generation_chromosomes(chromosomeNum)
    crossoverMutationNum = (1-cp)*chromosomeNum
    copyNum = cp*chromosomeNum
    mutationNum = p_m*chromosomeNum
    for ite in range(1,iteratorNum):
        Adaptability_allChromo = calAdaptability_chromosome(Y_allChromo)
        selectionProbability_allChromo = calSelectionProbability(Adaptability_allChromo,chromosomeNum)
        Y_allChromo = createGeneration(Y_allChromo,Adaptability_allChromo,selectionProbability_allChromo)
        max_Optimization_target = getmaxAdaptability(Adaptability_allChromo)
    print(max_Optimization_target)

Container_Placement_Based_On_GA()
