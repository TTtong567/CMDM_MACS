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
#X = np.zeros([Container_Num,10], dtype=int)
#for i in range(Container_Num):
    #X[i][np.random.randint(0,10)]=1
resource_used_EN = np.zeros([EN_Num,3], dtype=int)
resource_remaining_EN = np.zeros([EN_Num,3], dtype=int)
resource_utilization_EN = np.zeros([EN_Num,3], dtype=int)
load_EN = np.zeros(EN_Num)
load_differentiation_EN = np.zeros([EN_Num,EN_Num])
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
print('部署决策矩阵：\n', X)
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
print('边缘节点负载：\n', list(load_EN))
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


# 获取边缘节点上的容器集合
def getContainerList():
    ContainerList_total = []
    for i in range(EN_Num):
        ContainerList = []
        for j in range(Container_Num):
            if X[j][i] == 1:
                ContainerList.append(j)
        ContainerList_total.append(ContainerList)
    return ContainerList_total


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


# 计算资源均衡度
def getresourcebalancedegree(Z,EN_before,EN_after):
    resource_used_Matrix = np.zeros([EN_Num, 3], dtype=int)
    resource_remaining_Matrix = np.zeros([EN_Num, 3], dtype=int)
    resource_utilization_Matrix = np.zeros([EN_Num, 3], dtype=int)
    resource_uti_total_0 = resource_uti_total_1 = resource_uti_total_2 = 0
    resource_bal_total_0 = resource_bal_total_1 = resource_bal_total_2 = 0
    resource_remaining_total = 0
    migration_cost_total = 0
    for i in range(Container_Num):
        for j in range(EN_Num):
            if Z[i][j] == 1:
                resource_used_Matrix[j][0] += resource_Container[i][0]
                resource_used_Matrix[j][1] += resource_Container[i][1]
                resource_used_Matrix[j][2] += resource_Container[i][2]
                break
            else:
                continue
    resource_remaining_Matrix = resource_total_EN - resource_used_Matrix
    resource_utilization_Matrix = np.around(resource_used_Matrix / resource_total_EN, decimals=2)
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
    if (Z == X).all():
        migration_cost_total = 0
    else:
        migration_cost_total = data_trans_Container[i] / (1024 * Band[EN_before][EN_after] / 1000)
    Optimization_Target = (resource_balancing_total+resource_remaining_total+migration_cost_total)/3
    return Optimization_Target



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
            optimization_target_total = []
            mig_Container.append(j)
            resource_used_EN[i][0] -= resource_Container[j][0]
            resource_used_EN[i][1] -= resource_Container[j][1]
            resource_used_EN[i][2] -= resource_Container[j][2]
            Y[j][i] = 0
            for k in range(EN_Num):
                if k==i:
                    optimization_target = 100
                    optimization_target_total.append(optimization_target)
                else:
                    Y[j][k] = 1
                    optimization_target = getresourcebalancedegree(Y,i,k)
                    optimization_target_total.append(optimization_target)
                    Y[j][k] = 0
            max_min_optimization_target_list = np.argsort(optimization_target_total)
            temp = max_min_optimization_target_list[0]
            resource_used_EN[temp] += resource_Container[j]
            resource_remaining_EN[temp] -= resource_Container[j]
            Y[j][temp]=1
            if resource_used_EN[i][0]/resource_total_EN[i][0] < 0.6 and resource_used_EN[i][1]/resource_total_EN[i][1] < 0.9 and resource_used_EN[i][2]/resource_total_EN[i][2] < 0.9:
                break
            else:
                continue
        resource_remaining_EN[i] = resource_total_EN[i] - resource_used_EN[i]
        resource_utilization_EN[i] = np.around(resource_used_EN[i] / resource_total_EN[i], decimals=2)
    return mig_Container

migrated_ContainerList = Container_Selection()
#print("待迁移容器集合：\n",migrated_ContainerList)
print("迁移后节点资源利用率：\n",resource_utilization_EN)
print("迁移后节点剩余资源：\n",resource_remaining_EN)


def getfinalresourcebalancedegree(Z):
    resource_used_Matrix = np.zeros([EN_Num, 3], dtype=int)
    resource_remaining_Matrix = np.zeros([EN_Num, 3], dtype=int)
    resource_utilization_Matrix = np.zeros([EN_Num, 3], dtype=int)
    resource_uti_total_0 = resource_uti_total_1 = resource_uti_total_2 = 0
    resource_bal_total_0 = resource_bal_total_1 = resource_bal_total_2 = 0
    resource_remaining_total = 0
    migration_cost_total = 0
    for i in range(Container_Num):
        for j in range(EN_Num):
            if Z[i][j] == 1:
                resource_used_Matrix[j][0] += resource_Container[i][0]
                resource_used_Matrix[j][1] += resource_Container[i][1]
                resource_used_Matrix[j][2] += resource_Container[i][2]
                break
            else:
                continue
    resource_remaining_Matrix = resource_total_EN - resource_used_Matrix
    resource_utilization_Matrix = np.around(resource_used_Matrix / resource_total_EN, decimals=2)
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
    if (Z == X).all():
        migration_cost_total = 0
    else:
        for i in migrated_ContainerList:
            EN_before = getENbeforemigrating(i,X)
            EN_after = getENbeforemigrating(i,Z)
            migration_cost_total += (data_trans_Container[i]+resource_Container[i][0]/resource_total_EN[EN_before][0])/(1024*Band[EN_before][EN_after]/1000)
    Optimization_Target = (resource_balancing_total+resource_remaining_total+migration_cost_total)/3
    print(resource_balancing_total+resource_remaining_total)
    print(migration_cost_total)
    print(Optimization_Target)

print(Y)
getfinalresourcebalancedegree(Y)
