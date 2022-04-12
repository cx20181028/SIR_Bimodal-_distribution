# -*- coding:utf-8 -*-
import random

import networkx as nx
import  numpy as np
import EoN
import seaborn as sns
import matplotlib.pyplot as plt
# BA scale-free degree network
 # generalize BA network which has 20 nodes, m = 1---1000,12--1000,3
G = nx.random_graphs.barabasi_albert_graph(1000, 6)
print("平均聚类系数(average clustering): ", nx.average_clustering(G))

def node_status_update(G, node, beta, gamma):
    """
    更新节点状态
    :param G: 输入图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    if G.nodes[node]['status'] == 'I':
        p = random.random()

        if p < gamma:
            # print("I-sucess")
            # print(p)
            G.nodes[node]['status'] = 'R'
        # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)
    if G.nodes[node]['status'] == 'S':
        # 获取当前节点的邻居节点
        # 无向图：G.neighbors(node)
        # 有向图：G.predecessors(node)，前驱邻居节点，即指向该节点的节点；G.successors(node)，后继邻居节点，即该节点指向的节点。
        neighbors = list(G.neighbors(node))
        if len(neighbors) == 1:
            p = random.random()
            if p < beta:
                # print("S-sucess")
                # print(p)
                G.nodes[node]['status'] = 'I'

        elif len(neighbors) > 1:
            p = random.random()
            if p < beta * len(neighbors):
                # print("S-sucess")
                # print(p)
                G.nodes[node]['status'] = 'I'
def count_node(G):
    """
    计算当前图内各个状态节点的数目
    :param G: 输入图
    :return: 各个状态（S、I、R）的节点数目
    """
    s_num, i_num, r_num = 0, 0, 0
    for node in G:
        if G.nodes[node]['status'] == 'S':
            s_num += 1
        elif G.nodes[node]['status'] == 'I':
            i_num += 1
        else:
            r_num += 1
    return s_num, i_num, r_num
def SIR_Complex_Contagion():
    """
    更新节点状态
    :param G: 输入图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    Gnum=G.number_of_nodes()
    epoch=20
    #初始化节点状态为易感状态S
    for i in range(Gnum):
        G.nodes[i]['status']='S'

    #初始化初始节点i为感染状态，index=0
    G.nodes[0]['status']='I'
    Inum=[]
    pandq = np.arange(0, 1, 0.025)
    print(pandq)
    iterations = 500  # run 5 simulations

    phase = []
    for i in range(len(pandq) - 1, -1, -1):
        p = []
        for j in range(len(pandq)):
            Rn = []
            #simulation iteration
            for counter in range(iterations):
                #simulate SIR
                for t in range(0, epoch):
                    for node in range(0,Gnum):
                        node_status_update(G,node, pandq[j], pandq[i])
                        s_num, i_num, r_num = count_node(G)
                        Inum.append(i_num)
                    Rn.append(max(Inum))
                p.append(float(np.mean(Rn)) / G.number_of_nodes())
    print(phase)
    t = np.arange(0, 1, 0.025)
    sns.heatmap(phase, cmap='gist_heat_r', linewidths=0.01, linecolor="black",
                xticklabels=np.arange(0, 1, 0.025),
                # x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
                yticklabels=sorted(t, reverse=True),  # y轴方向刻度标签开关、同x轴
                )
    plt.title('phase diagram')
    plt.xlabel('spread-p  x10^-2')
    plt.ylabel('recover-q  x10^-2')
    plt.show()


def phase_diagram_infected():
    pandq=np.arange(0,1,0.025)
    print(pandq)
    iterations =500  # run 5 simulations
    initial_node=[0]

    phase=[]
    for i in range(len(pandq)-1,-1,-1):
        p=[]
        for j in range(len(pandq)):
            Rn = []
            for counter in range(iterations):
                t, S, I, R = EoN.fast_SIR(G, pandq[j], pandq[i], initial_infecteds =initial_node)
                Rn.append(max(I))
            p.append(float(np.mean(Rn))/G.number_of_nodes())
        phase.append(p)
    print(phase)
    t=np.arange(0,1,0.025)

    sns.heatmap(phase, cmap='gist_heat_r',linewidths=0.01,linecolor="black",
                xticklabels=np.arange(0,1,0.025),
                # x轴方向刻度标签开关、赋值，可选“auto”, bool, list-like（传入列表）, or int,
                yticklabels= sorted(t,  reverse=True),  # y轴方向刻度标签开关、同x轴

                )
    plt.title('phase diagram')
    plt.xlabel('spread-p  ')
    plt.ylabel('recover-q ')
    plt.show()
def caculate_triangle(G):
    '''
只需要计算三角形，请使用：
 import networkx as nx tri=nx.triangles(g)
但是如果需要知道具有三角形（三元）关系的边列表，请使用
all_cliques= nx.enumerate_all_cliques(g)
    :param G:
    :return:
    '''
    tri = nx.triangles(G)
    print("三角形个数", nx.triangles(G))
    triangle_num = nx.triangles(G)
    print(sum(triangle_num.values()))
    print(type(nx.triangles(G)))
def phase_diagram_epidemic_range():


    return
caculate_triangle(G)
#Ephase_diagram_infected()
SIR_Complex_Contagion()