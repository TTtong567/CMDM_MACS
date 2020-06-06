#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
import os
import random
import re
import math
import numpy as np

Container_Num = 180
#resource_total_EN =np.array([[32, np.random.randint(16,65), np.random.randint(100,400)]for i in range(10)])
#resource_Container =np.array([[np.random.randint(1,3),np.random.randint(1,3),np.random.randint(2,15)]for i in range(Container_Num)])
#data_trans_Container = np.random.randint(100,500,Container_Num)
#Band = np.triu(np.random.randint(100,300,100).reshape(10, 10))
#Band += Band.T - np.diag(Band.diagonal())
#Band = Band - np.diag(np.diag(Band))
#X = np.zeros([Container_Num,10], dtype=int)
#for i in range(Container_Num):
    #X[i][np.random.randint(0,10)]=1
resource_used_EN = np.zeros([10,3], dtype=int)
resource_remaining_EN = np.zeros([10,3], dtype=int)
resource_utilization_EN = np.zeros([10,3], dtype=int)
load_EN = np.zeros(10)
load_differentiation_EN = np.zeros([10,10])
resource_total_EN = np.loadtxt('D:\ACS5\Resource_total_EN.txt', dtype='int', delimiter=',')
resource_Container = np.loadtxt('D:\ACS5\Resource_Container.txt', dtype='int', delimiter=',')
data_trans_Container = np.loadtxt('D:\ACS5\data_trans_Container.txt', dtype='int', delimiter=',')
Band = np.loadtxt('D:\ACS5\Band.txt', dtype='int', delimiter=',')
X = np.loadtxt('D:\ACS5\X.txt', dtype='int', delimiter=',')
print('边缘节点资源总量：\n', resource_total_EN)
print('容器资源需求：\n', resource_Container)
print('容器数据传输总量：\n', data_trans_Container)
print('边缘节点间带宽：\n', Band)
print('部署决策矩阵：\n', X)
Y = np.copy(X)
for i in range(Container_Num):
    for j in range(10):
        if X[i][j] == 1:
            resource_used_EN[j][0] += resource_Container[i][0]
            resource_used_EN[j][1] += resource_Container[i][1]
            resource_used_EN[j][2] += resource_Container[i][2]
            break
        else:
            continue
resource_remaining_EN = resource_total_EN-resource_used_EN
resource_utilization_EN = np.around(resource_used_EN/resource_total_EN, decimals=2)
for i in range(10):
    load_EN[i] = (resource_utilization_EN[i][0]+resource_utilization_EN[i][1]+resource_utilization_EN[i][2])/3
for i in range(10):
    for j in range(10):
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
    for i in range(10):
        for j in range(10):
            if load_differentiation_EN[i][j] >= 2:
                temp.append(i)
                break
            else:
                continue
    for i in range(10):
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
    for i in range(10):
        ContainerList = []
        for j in range(Container_Num):
            if X[j][i] == 1:
                ContainerList.append(j)
        ContainerList_total.append(ContainerList)
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
            mig_quantity = random.random()
            for k in range(3):
                mig_number += (resource_utilization_EN[i][k] / sum_resource_utilization_EN)/math.sqrt(math.pow((resource_utilization_EN[i][k]-resource_Container[j][k]/resource_total_EN[i][k]),2))
            migration_probability = mig_time
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
print("待迁移容器集合：\n",migrated_ContainerList)
print("迁移后节点资源利用率：\n",resource_utilization_EN)
print("迁移后节点剩余资源：\n",resource_remaining_EN)


# 获取待迁移容器所在边缘节点
def getENbeforemigrating(i,X):
    EN_beforemigrating = 0
    for j in range(10):
        if X[i][j]==1:
            EN_beforemigrating =j
            break
    return EN_beforemigrating


# 获取容器的有效边缘节点集合
def getavailableENSet(l):
    available_EN = []
    jj = getENbeforemigrating(l,X)
    for j in range(10):
        if (resource_used_EN[j][0] + resource_Container[l][0])/resource_total_EN[j][0] < 0.6 and (resource_used_EN[j][1] + resource_Container[l][1])/resource_total_EN[j][1] < 0.9 and (resource_used_EN[j][2] - resource_Container[l][2])/resource_total_EN[j][2] < 0.9 and j != jj:
            available_EN.append(j)
    return available_EN


# 初始化启发式信息
def initheuristicMatrix(HM):
    for i in migrated_ContainerList:
        jj = getENbeforemigrating(i,X)
        for j in range(10):
            if (resource_remaining_EN[j][0] - resource_Container[i][0])/resource_total_EN[j][0] > 0.4 and (resource_remaining_EN[j][1] - resource_Container[i][1])/resource_total_EN[j][1] > 0.1 and (resource_remaining_EN[j][2] - resource_Container[i][2])/resource_total_EN[j][2] > 0.1:
                heuristic_info_1 =((resource_remaining_EN[j][0] - resource_Container[i][0])/resource_total_EN[j][0]+(resource_remaining_EN[j][1] - resource_Container[i][1])/resource_total_EN[j][1]+(resource_remaining_EN[j][2] - resource_Container[i][2])/resource_total_EN[j][2])/\
                                  (abs((resource_remaining_EN[j][1] - resource_Container[i][1])/resource_total_EN[j][1]-(resource_remaining_EN[j][0] - resource_Container[i][0])/resource_total_EN[j][0])+\
                                  abs((resource_remaining_EN[j][2] - resource_Container[i][2])/resource_total_EN[j][2]-(resource_remaining_EN[j][0] - resource_Container[i][0])/resource_total_EN[j][0])+\
                                  abs((resource_remaining_EN[j][2] - resource_Container[i][2])/resource_total_EN[j][2]-(resource_remaining_EN[j][1] - resource_Container[i][1])/resource_total_EN[j][1])+0.1)
            else:
                heuristic_info_1 = 0
            if j!=jj:
                heuristic_info_2 = Band[jj][j]
            else:
                heuristic_info_2 = 0
            HM[i][j] = heuristic_info_1*heuristic_info_2
    heuristicMatrix = np.around(HM, decimals=2)


# 计算资源均衡度
def getresourcebalancedegree(Z):
    resource_used_Matrix = np.zeros([10, 3], dtype=int)
    resource_remaining_Matrix = np.zeros([10, 3], dtype=int)
    resource_utilization_Matrix = np.zeros([10, 3], dtype=int)
    resource_uti_total_0 = resource_uti_total_1 = resource_uti_total_2 = 0
    resource_bal_total_0 = resource_bal_total_1 = resource_bal_total_2 = 0
    resource_remaining_total = 0
    migration_cost_total = 0
    for i in range(Container_Num):
        for j in range(10):
            if Z[i][j] == 1:
                resource_used_Matrix[j][0] += resource_Container[i][0]
                resource_used_Matrix[j][1] += resource_Container[i][1]
                resource_used_Matrix[j][2] += resource_Container[i][2]
                break
            else:
                continue
    resource_remaining_Matrix = resource_total_EN - resource_used_Matrix
    resource_utilization_Matrix = np.around(resource_used_Matrix / resource_total_EN, decimals=2)
    for i in range(10):
        resource_uti_total_0 += resource_utilization_Matrix[i][0]
        resource_uti_total_1 += resource_utilization_Matrix[i][1]
        resource_uti_total_2 += resource_utilization_Matrix[i][2]
    resource_average_0 = resource_uti_total_0 / 10
    resource_average_1 = resource_uti_total_1 / 10
    resource_average_2 = resource_uti_total_2 / 10
    for i in range(10):
        resource_bal_total_0 += (resource_utilization_Matrix[i][0] - resource_average_0) ** 2 / 10
        resource_bal_total_1 += (resource_utilization_Matrix[i][1] - resource_average_1) ** 2 / 10
        resource_bal_total_2 += (resource_utilization_Matrix[i][2] - resource_average_2) ** 2 / 10
    resource_balancing_total = resource_bal_total_0 + resource_bal_total_1 + resource_bal_total_2
    for i in range(10):
        resource_remaining_total +=(abs(resource_utilization_Matrix[i][1]-resource_utilization_Matrix[i][0])+abs(resource_utilization_Matrix[i][2]-resource_utilization_Matrix[i][0])+abs(resource_utilization_Matrix[i][2]-resource_utilization_Matrix[i][1]))
    if (Z == X).all():
        migration_cost_total = 0
    else:
        for i in migrated_ContainerList:
            EN_before = getENbeforemigrating(i,X)
            EN_after = getENbeforemigrating(i,Z)
            migration_cost_total += (data_trans_Container[i] + resource_Container[i][0] / resource_total_EN[EN_before][0]) / (1024 * Band[EN_before][EN_after] / 1000)
    Optimization_Target = (resource_balancing_total+resource_remaining_total+migration_cost_total)/3
    return Optimization_Target


# 全局信息素更新
def updateGlobalPheromoneMatrix(Y_oneIt, PM, Optimization_target_oneIt,gl_q):
    mintarget_AntIndex = 0
    min_Optimization_target = Optimization_target_oneIt[0]
    for i in range(len(Optimization_target_oneIt)):
        if Optimization_target_oneIt[i]<min_Optimization_target:
            mintarget_AntIndex = i
            min_Optimization_target = Optimization_target_oneIt[i]
    for i in migrated_ContainerList:
        for j in range(10):
            if Y_oneIt[mintarget_AntIndex][i][j] ==1:
                #PM[i][j] = (1-gl_q) * PM[i][j] + gl_q/Optimization_target_oneIt[mintarget_AntIndex]
                PM[i][j] += gl_q / Optimization_target_oneIt[mintarget_AntIndex]
                break
            else:
                continue
    return min_Optimization_target


# 获取容器迁移后的目标边缘节点
def ENafterContainermigrated(tomigcontainer,PM,HM,al,be,q0):
    available_EN = getavailableENSet(tomigcontainer)
    q = random.random()
    if q <= q0:
        max_min_list = np.argsort(-(PM[tomigcontainer] ** al) * (HM[tomigcontainer] ** be))
        for temp in max_min_list:
            if temp in available_EN:
                EN_selected = temp
                break
            else:
                continue
    else:
        wheel = random.random()
        probability_total = 0
        phe_heu_all = []
        phe_heu_total = 0
        for j in range(10):
            phe_heu = (PM[tomigcontainer][j] ** al) * (HM[tomigcontainer][j] ** be)
            phe_heu_total += phe_heu
            phe_heu_all.append(phe_heu)
        for i in range(len(phe_heu_all)):
            phe_heu_all[i] = phe_heu_all[i] / phe_heu_total
        for k in range(len(phe_heu_all)):
            probability_total += phe_heu_all[k]
            if probability_total >= wheel:
                EN_selected = k
                break
            else:
                continue
    return EN_selected


# 容器部署
def Container_Placement_Based_On_MACS():
    mig_Container_number = len(migrated_ContainerList)
    iteratorNum =200  # 迭代次数
    antsNum = 10      # 蚂蚁个数
    alpha = 1         # 信息启发因子
    beta = 2          # 期望启发因子
    local_q = 0.1     # 信息素局部蒸发
    global_q = 0.85      # 信息素强度
    q_0 = 0.1
    pheromoneMatrix = np.ones([Container_Num,10])
    heuristicMatrix = np.zeros([Container_Num,10],dtype=int)
    for ite in range(iteratorNum):
        Y_allAnt = []  # 一次迭代 -> 10个决策数组
        Optimization_target_allAnt = []   # 一次迭代 -> 10个最优化目标值
        global Y
        copy_Y = np.copy(Y)
        for ant in range(antsNum):
            global resource_used_EN
            global resource_remaining_EN
            copy_resource_used_EN = np.copy(resource_used_EN)
            copy_resource_remaining_EN = np.copy(resource_remaining_EN)
            for i in migrated_ContainerList:
                initheuristicMatrix(heuristicMatrix)
                temp = ENafterContainermigrated(i, pheromoneMatrix, heuristicMatrix, alpha, beta, q_0)
                Y[i][temp] = 1
                resource_used_EN[temp] += resource_Container[i]
                resource_remaining_EN[temp] -= resource_Container[i]
                pheromoneMatrix[i][temp] *= (1-local_q)        # 局部信息素衰减
            resource_used_EN = np.copy(copy_resource_used_EN)
            resource_remaining_EN = np.copy(copy_resource_remaining_EN)
            Y_allAnt.append(Y)
            Y = np.copy(copy_Y)
        for i in range(antsNum):
            optimization_value_oneAnt = getresourcebalancedegree(Y_allAnt[i])
            #print("optimization_value_oneAnt:",optimization_value_oneAnt)
            Optimization_target_allAnt.append(optimization_value_oneAnt)
        #print(Optimization_target_allAnt)
        min_Optimization_target_oneAnt = updateGlobalPheromoneMatrix(Y_allAnt, pheromoneMatrix, Optimization_target_allAnt, global_q)
        #print("一次迭代后的信息素矩阵：\n", pheromoneMatrix)
        #print(min_Optimization_target_oneAnt)
    return Y_allAnt[-1]
Y_bestAnt = Container_Placement_Based_On_MACS()



# 负载-迁移代价计算
def calresource_cost(Z):
    resource_used_Matrix = np.zeros([10, 3], dtype=int)
    resource_remaining_Matrix = np.zeros([10, 3], dtype=int)
    resource_utilization_Matrix = np.zeros([10, 3], dtype=int)
    resource_uti_total_0 = resource_uti_total_1 = resource_uti_total_2 = 0
    resource_bal_total_0 = resource_bal_total_1 = resource_bal_total_2 = 0
    resource_remaining_total = 0
    migration_cost_total = 0
    for i in range(Container_Num):
        for j in range(10):
            if Z[i][j] == 1:
                resource_used_Matrix[j][0] += resource_Container[i][0]
                resource_used_Matrix[j][1] += resource_Container[i][1]
                resource_used_Matrix[j][2] += resource_Container[i][2]
                break
            else:
                continue
    resource_remaining_Matrix = resource_total_EN - resource_used_Matrix
    resource_utilization_Matrix = np.around(resource_used_Matrix / resource_total_EN, decimals=2)
    for i in range(10):
        resource_uti_total_0 += resource_utilization_Matrix[i][0]
        resource_uti_total_1 += resource_utilization_Matrix[i][1]
        resource_uti_total_2 += resource_utilization_Matrix[i][2]
    resource_average_0 = resource_uti_total_0 / 10
    resource_average_1 = resource_uti_total_1 / 10
    resource_average_2 = resource_uti_total_2 / 10
    for i in range(10):
        resource_bal_total_0 += (resource_utilization_Matrix[i][0] - resource_average_0) ** 2 / 10
        resource_bal_total_1 += (resource_utilization_Matrix[i][1] - resource_average_1) ** 2 / 10
        resource_bal_total_2 += (resource_utilization_Matrix[i][2] - resource_average_2) ** 2 / 10
    resource_balancing_total = resource_bal_total_0 + resource_bal_total_1 + resource_bal_total_2
    for i in range(10):
        resource_remaining_total +=(abs(resource_utilization_Matrix[i][1]-resource_utilization_Matrix[i][0])+abs(resource_utilization_Matrix[i][2]-resource_utilization_Matrix[i][0])+abs(resource_utilization_Matrix[i][2]-resource_utilization_Matrix[i][1]))
    if (Z == X).all():
        migration_cost_total = 0
    else:
        for i in migrated_ContainerList:
            EN_before = getENbeforemigrating(i,X)
            EN_after = getENbeforemigrating(i,Z)
            migration_cost_total += (data_trans_Container[i] + resource_Container[i][0] / resource_total_EN[EN_before][0]) / (1024 * Band[EN_before][EN_after] / 1000)
    print(resource_balancing_total + resource_remaining_total)
    print(migration_cost_total)
    print((resource_balancing_total + resource_remaining_total + migration_cost_total) / 3)

calresource_cost(Y_bestAnt)