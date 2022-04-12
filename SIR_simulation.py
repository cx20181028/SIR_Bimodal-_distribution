import random

import EoN
import matplotlib.pyplot as plt
import networkx as nx
import  numpy as np
from collections import Counter
N = 3000
iterations = 1000

#ln3000=8 ln50000=10
def constructRR():#生成N个节点的网络，每次连接2对节点，并进行是否为全连接网络的判断，如果是的话，那就计算出该网络下的βc
    while True:
        #G=nx.barabasi_albert_graph(N,2)
        G=nx.erdos_renyi_graph(N,0.003)
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
def SIMULATION_SIR(G, tau, gamma,  mark):
    initial_node = [90]
    Rt = []
    for x in range(2000):
        t, S, I, R = EoN.fast_SIR(G, tau, 1, initial_infecteds=initial_node)
        Rt.append(R[-1]/G.number_of_nodes())
    print(Rt)
    plt.figure()
    plt.hist(Rt)
    plt.title("R态占比分布")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.show()
def sum_dict(a,b):
  temp = dict()
  # python3,dict_keys类似set； | 并集
  for key in a.keys()| b.keys():
    temp[key] = sum([d.get(key, 0) for d in (a, b)])
  return temp
import collections
def WFS(G,node,beta):
    queue = collections.deque()
    queue.append(90)
    while len(queue) != 0:
     nodes   = queue.popleft()
     neighbors = list(G.neighbors(nodes))
     for neighbor in neighbors:
        # 邻居节点中存在 感染者(I)，则该节点有概率被感染为 感染者(I)
            p = (random.random()*10)/10
            if p < 1-beta:
                G.remove_edge(node, neighbor)
            else:
                queue.append(neighbor)
def Component_size_distribute(G, beta ):
    #simutate percolation
     # 产生0-1浮点数

    node=G.nodes
    Gnum=G.number_of_nodes()
    edges=list(G.edges)
    x = [i for i in range(len(edges))]
    random.shuffle(x)
    for bond in x:
        occupied = random.random()
        #print("occupied %s"%(str(occupied)))
        if occupied<1-beta:
            G.remove_edge(edges[bond][0],edges[bond][1])
            #print("sucess")
            #print(bond)
    '''
    m = []

    for i in range(G.number_of_nodes()):
        m.append(1)
    m[90] = 0
    Gnum = G.number_of_nodes()
    v = [-1 for i in range(G.number_of_nodes())]

    queue = []
    queue.append(90)

    while len(queue) != 0:
        nodes = queue.pop(0)

        neighbors = list(G.neighbors(nodes))
        for neighbor in neighbors:
            if m[neighbor]==1:

                p = (random.random()*10)/10
                if p < 1 - beta:
                        G.remove_edge(nodes, neighbor)
                        m[neighbor] =1

                m[neighbor]=0
                queue.append(neighbor)

     '''
    gen = [c for c in list(nx.connected_components(G))]
    lenghs=[len(c) for c in gen]#统计每个connected-component的大小
    lenghs.sort(reverse=True)
    t=nx.node_connected_component(G, 90)
    maxG_node_num=len(t)
    results=Counter(lenghs)#统计每个SIZE的component各有多少个
    #mydict=dict(results)
    return  maxG_node_num,results

if __name__ == '__main__':
#0是小于，1是等于，2是大于βC
    mark=0
    ''''''
    #iterations = 100  # run 5 simulations
    #tau = 0.1  # transmission rate
    #gamma = 0.05  # recovery rate
    #rho = 0.05  # random fraction initially infected，num=num_of_nodes*rho
    G,beta_c=constructRR()
    if mark==2:
        beta = beta_c*1.35
        iter = 2000
        gamma=1
        print(beta)
        initial_node = [90]
        Rt = []
        for x in range(iter):
            t, S, I, R = EoN.fast_SIR(G, beta, gamma, initial_infecteds=initial_node)
            Rt.append(R[-1] )
        print(G.number_of_nodes())
        print(Rt)
        print(Counter(Rt))
        plt.figure()
        plt.hist(Rt,color="r")
        plt.title("R态占比分布")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.show()
        #Evolution_SIR(G, [1], beta,gamma)
        #SIMULATION_SIR(G, beta, gamma, mark)
        print("ttt")
        totald=[]
        totalmax=[]

        for i in range(iter):
            G1 = nx.Graph(G)
            max, mydict = Component_size_distribute(G1, beta)
            totalmax.append(max)
            totald.append(mydict)

        plt.figure()
        print(totalmax)
        plt.hist(totalmax)
        plt.title("Percolation-R态占比分布")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.show()
        from functools import reduce
        emerge_d = reduce(sum_dict, totald)
        emerge_d = dict(sorted(emerge_d.items(), key=lambda x: x[0]))
        sizes = []
        size_num = []
        total_s = 0
        for key, value in emerge_d.items():
            total_s += (emerge_d[key] / iter)
        for key, value in emerge_d.items():
            size_num.append(emerge_d[key] / (total_s * iter))
            sizes.append((key))
        import  math
        #size_num = [math.log10(i) for i in size_num]
        plt.scatter(sizes, size_num)
        plt.xlabel('SIZE s')
        plt.ylabel('p(s)')
        plt.title("component size 分布")
        plt.show()
        '''
        for i in range(iter):
            G1 = nx.Graph(G)
            mydict,max = Component_size_distribute(G1, beta)
            totald.append(mydict)
        from functools import reduce
        emerge_d = reduce(sum_dict, totald)
        emerge_d = dict(sorted(emerge_d.items(), key=lambda x: x[0]))
        sizes = []
        size_num = []
        total_s=0
        for key, value in emerge_d.items():
            total_s+=(emerge_d[key]/iter)
        for key, value in emerge_d.items():
            size_num.append(emerge_d[key] / (total_s*iter))
            sizes.append((key))
        plt.scatter(sizes, size_num)
        plt.xlabel('SIZE s')
        plt.ylabel('p(s)')
        plt.show()

        
        sizes,size_num=Component_size_distribute(G, beta)
        plt.plot(sizes,size_num,label="distribution of size of connected component")
        plt.title("β大于βC")
        plt.xlabel("S")
        plt.ylabel("S-number/N")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.legend(loc="upper right")
        plt.show()
       '''
    elif mark==0:
        uper=beta_c*0.5
        beta =beta_c*0.9
        gamma=1
        #mydict=Component_size_distribute(G,beta)
        iter = 2000
        gamma = 1
        print(beta)
        initial_node = [90]
        Rt = []
        for x in range(iter):
            t, S, I, R = EoN.fast_SIR(G, beta, gamma, initial_infecteds=initial_node)
            Rt.append(R[-1])
        print(G.number_of_nodes())
        print(Rt)
        print(Counter(Rt))
        plt.figure()
        plt.hist(Rt,color="r")
        plt.title("R态占比分布")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.show()
        # Evolution_SIR(G, [1], beta,gamma)
        # SIMULATION_SIR(G, beta, gamma, mark)
        print("ttt")
        totald = []
        totalmax = []

        for i in range(iter):
            G1 = nx.Graph(G)
            max, mydict = Component_size_distribute(G1, beta)
            totalmax.append(max)
            totald.append(mydict)

        plt.figure()
        print(totalmax)
        plt.hist(totalmax)
        plt.title("Percolation-R态占比分布")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.show()
        from functools import reduce

        emerge_d = reduce(sum_dict, totald)
        emerge_d = dict(sorted(emerge_d.items(), key=lambda x: x[0]))
        sizes = []
        size_num = []
        total_s = 0
        for key, value in emerge_d.items():
            total_s += (emerge_d[key] / iter)
        for key, value in emerge_d.items():
            size_num.append(emerge_d[key] / (total_s * iter))
            sizes.append((key))
        plt.scatter(sizes, size_num)
        plt.title("component size 分布")
        plt.xlabel('SIZE s')
        plt.ylabel('p(s)')
        plt.show()
        '''
        plt.hist(size_num)
        plt.plot(sizes, size_num, label="distribution of size of connected component")
        plt.title("β小于βC")
        plt.xlabel("S")
        plt.ylabel("S-number/N")
        plt.legend(loc="upper right")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签        
        '''


    elif mark==1:

        beta =beta_c
        gamma=beta

        iter = 1000
        gamma = 1
        print(beta)
        initial_node = [90]
        Rt = []
        for x in range(iter):
            G1 = nx.Graph(G)
            t, S, I, R = EoN.fast_SIR(G1, beta, gamma, initial_infecteds=initial_node)
            Rt.append(R[-1])
        print(G.number_of_nodes())
        print(Rt)
        print(Counter(Rt))
        plt.figure()
        plt.hist(Rt,color="r")
        plt.title("R态占比分布")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.show()
        # Evolution_SIR(G, [1], beta,gamma)
        # SIMULATION_SIR(G, beta, gamma, mark)
        print("ttt")
        totald = []
        totalmax = []

        for i in range(iter):
            G1 = nx.Graph(G)
            print(i)
            max, mydict = Component_size_distribute(G1, beta)
            totalmax.append(max)
            totald.append(mydict)

        plt.figure()
        print(totalmax)
        plt.hist(totalmax)
        plt.title("Percolation-R态占比分布")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.show()
        from functools import reduce

        emerge_d = reduce(sum_dict, totald)
        emerge_d = dict(sorted(emerge_d.items(), key=lambda x: x[0]))
        sizes = []
        size_num = []
        total_s = 0
        for key, value in emerge_d.items():
            total_s += (emerge_d[key] / iter)
        for key, value in emerge_d.items():
            size_num.append(emerge_d[key] / (total_s * iter))
            sizes.append((key))
        import  math
        size_num=[math.log10(i)  for  i in size_num]
        plt.scatter(sizes, size_num)
        plt.title("component size 分布")
        plt.xlabel('SIZE s')
        plt.ylabel('p(s)')
        parameter = np.polyfit(sizes, size_num, 1)
        p = np.poly1d(parameter)
        plt.plot(sizes, p(sizes), color='g')
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
        plt.show()
        '''
        plt.hist(size_num)
        plt.show()
        plt.plot(sizes, size_num, label="distribution of size of connected component")
        plt.title("β等于βC")
        plt.xlabel("S/N")
        plt.ylabel("S-number")
        plt.legend(loc="upper right")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签

        plt.show()        
        '''
