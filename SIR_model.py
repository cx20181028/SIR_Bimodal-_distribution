# EoN官网demo
import random
import EoN
import matplotlib.pyplot as plt
import  numpy as np
import networkx as nx
N=400
def constructRR():
    while True:
        G=nx.barabasi_albert_graph(N, 5)
        if nx.is_connected(G):
            degree=nx.degree_centrality(G)
            degreelist=list(degree.values())
            degreelists= [int(i * (N-1)) for i in  degreelist]

            degree_avg=np.mean(degreelists)
            degree_squar=[int(i*i) for i in degreelists]
            degree_squar_avg=np.mean(degree_squar)
            beta_c=(degree_avg)/(degree_squar_avg-degree_avg)
            print(beta_c)
            print("success")
            break
    return  G,beta_c
def node_status_uodate(G, nodeIndex, beta, gamma):
    """
    更新节点状态
    :param G: 输入图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    # 如果当前节点状态为 感染者(I) 有概率gamma变为 免疫者(R)
    if G.nodes[nodeIndex]['status'] == 'I':
        p = random.random()

        if p < gamma:
            #print("I-sucess")
            #print(p)
            G.nodes[nodeIndex]['status'] = 'R'
    # 如果当前节点状态为 易感染者(S) 有概率beta变为 感染者(I)
    if G.nodes[nodeIndex]['status'] == 'S':
        # 获取当前节点的邻居节点
        # 无向图：G.neighbors(node)
        # 有向图：G.predecessors(node)，前驱邻居节点，即指向该节点的节点；G.successors(node)，后继邻居节点，即该节点指向的节点。
        neighbors = list(G.neighbors(nodeIndex))
        # 对当前节点的邻居节点进行遍历
        for neighbor in neighbors:
            # 邻居节点中存在 感染者(I)，则该节点有概率被感染为 感染者(I)
            if G.nodes[neighbor]['status'] == 'I':
                p = random.random()
                if p < beta:
                    #print("S-sucess")
                    #print(p)
                    G.nodes[nodeIndex]['status'] = 'I'
                    break
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
def SIR(G, beta, gamma,mark):
    """
    更新节点状态
    :param G: 输入图
    :param node: 节点序数
    :param beta: 感染率
    :param gamma: 免疫率
    """
    Gnum=G.number_of_nodes()
    epoch=200
    t=100
    #初始化节点状态为易感状态S
    for i in range(Gnum):
        G.nodes[i]['status']='S'

    #初始化初始节点i为感染状态，index=0
    G.nodes[0]['status']='I'

    Snum=[]
    Rnum=[]
    Inum=[]
    Snum.append(Gnum-1)
    Rnum.append(0)
    Inum.append(1)
    for tt in range(t):
        for i in range(0,epoch):
            for node in range(0,Gnum):
              node_status_uodate(G, node, beta, gamma)

            if i+1==epoch:
                s_num, i_num, r_num = count_node(G)
                Snum.append(s_num)
                Rnum.append(r_num)
                Inum.append(i_num)


    t=[i for i in np.arange(0,t+1,1)]
    plt.plot(t,Snum,label="S-number")
    plt.plot(t,Rnum,label="R-number")
    plt.plot(t,Inum,label="I-number")
    plt.xlabel("t")
    plt.legend(loc="upper right")
    # 设置纵轴标签
    plt.ylabel("number")
    if mark==0:

        plt.title("β小于βC")
    elif mark==1:
        plt.title("β大于βC")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.show()

import networkx as nx
import random
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

"""
Function
----------
This function aims to achieve the simulation of SIR Model
Parameters
----------
1. G : Input Graph
2. Infected : The ndarray including the nodes infected at first
3. Gamma: Transmission probability
4. Beta: Recovery probability
5. Epoch : The iterations in total
6. Drawing: Whether to draw the pic in this function
"""
def Evolution_SIR(G, Infected, Gamma , Beta , Epoch = 100, Drawing = True):

    # Get the number of the nodes
    Node_num = len(G.nodes)

    # Initialize the state of the nodes, -1 denotes a recovered node, 1 denotes a suspetible node while 0 denotes an infected node
    Node_state = np.ones(Node_num).astype(np.int8)
    Node_state[Infected] = 0

    # Get the Adjacent matrix of the Graph
    A = np.array(nx.adjacency_matrix(G).astype(np.int8).todense())

    print("3.Adjacent Matrix is prepared!")

    x_label = [0]
    Infected_label = [len(Infected)]
    Recovery_label = [0]

    ### Problem describe
    for i in range(1, Epoch):

        x_label.append(i)
        # Fisrt Control the recovery of the Node Infected

        # Deep copy of Node_state
        Tmp_Node_state = Node_state.copy()
        zero_pos = np.array(np.where(Tmp_Node_state == 0)[0])
        one_pos = np.array(np.where(Tmp_Node_state == 1)[0])

        Tmp_Node_state[zero_pos] = 1
        Tmp_Node_state[one_pos] = 0

        x = np.random.rand(Node_num)
        x = x * Tmp_Node_state
        Recovry_iter = np.array(np.intersect1d(np.where(x < Beta)[0], np.where(x > 0)[0]))

        Recovery_label.append(len(Recovry_iter))

        # Set the recovery node state to -1
        Node_state[Recovry_iter] = -1

        Infected = np.setdiff1d(Infected, Recovry_iter)

        Tmp = Infected.copy()

        # Second Control the Node Infected
        for infected_index in Infected:
            # Generate the array containing all the random seed's state
            y = np.random.rand(Node_num)
            y = y * Node_state
            y = y * A[infected_index]

            # Get the state of this iter
            Infected_iter = np.array(np.intersect1d(np.where(y < Gamma)[0],np.where(y > 0)[0]))

            # Refresh the spreading in real time
            Node_state[Infected_iter] = 0

            # Refresh the state of the nodes
            Tmp = np.union1d(Infected_iter, Tmp)

        Infected = Tmp.copy()
        Infected_label.append(len(Infected))

    if Drawing == True:
        plt.xlabel("Iteration")
        plt.ylabel("Number")
        plt.plot(x_label,Recovery_label,label="Recovered")
        plt.plot(x_label,Infected_label,label="Infected")

        #ax = plt.gca()
        #ax.xaxis.set_major_locator(MultipleLocator(1))
       # plt.xlim(-0.5, Epoch)

        plt.legend()
        #plt.savefig("Output/"+str(G.name)+"/output_SIS.jpg")
        plt.show()

    return Infected_label, Recovery_label
G,beta_c=constructRR()

mark=1
''''''

if mark==1:
    low=beta_c+0.005
    beta = beta_c*2
    gamma=1
    #Evolution_SIR(G, [1], beta,gamma)
    SIR(G, beta=beta, gamma=gamma,mark=mark)
elif mark==0:
    uper=beta_c-0.005
    beta = beta_c*0.5
    gamma=1
    SIR(G, beta=beta, gamma=gamma,mark=mark)

