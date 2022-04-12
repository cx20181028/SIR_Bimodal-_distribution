import random

import networkx as nx
import  numpy as np
import EoN
import matplotlib.pyplot as plt
import  collections
# BA scale-free degree network
 # generalize BA network which has 20 nodes, m = 1---1000,12--1000,3
N=1000
#G = nx.random_graphs.barabasi_albert_graph(N, 6)
G=nx.erdos_renyi_graph(N,0.4)
degree = nx.degree_centrality(G)
degreelist = list(degree.values())
degreelists = [int(i * (N - 1)) for i in degreelist]

degree_avg = np.mean(degreelists)
degree_squar = [int(i * i) for i in degreelists]
degree_squar_avg = np.mean(degree_squar)
beta_c = (degree_avg) / (degree_squar_avg - degree_avg)
print(beta_c)
print("success")
print("平均聚类系数(average clustering): ", nx.average_clustering(G))
def phase_diagram_infected():
    pandq=np.arange(0,1,0.025)
    print(pandq)
    iterations =1000  # run 5 simulations
    initial_node=[0]
    gamma=beta_c
    #rates=[0.002,0.004,0.006,0.008,0.01]
    rates = [0.1, 0.2, 0.4, 0.6, 1]
    #rates = [1, 2, 4, 6, 10]
    #rates = [0.012, 0.014, 0.016, 0.018, 0.02]
    character=['o','-','*','s','x']
    color=['r','b','y','c','g']
    i=0
    for rate in rates:
        Rt=[]
        for counter in range(iterations):
                    t, S, I, R = EoN.fast_SIR(G,gamma ,gamma*rate , initial_infecteds =initial_node)
                    Rt.append(R[-1])

        print(type(Rt))
        t=dict(collections.Counter(Rt))
        t=dict(sorted(t.items(), key=lambda x: x[0]))
        print(dict(t))
        x=[]
        y=[]
        for key,value in t.items():
            x.append(key)
            y.append(value)
        bizhi=1/rate
        #titles="β/γ"+str(bizhi)
        titles =  "β="+str(rate)
        plt.plot(x, y, label=titles,color=color[i])
        i=i+1
    plt.title("R态分布")
    plt.xlabel("size")
    plt.ylabel("frequency")
    plt.legend(loc="upper right")
    #plt.figure()
    #plt.hist(Rt)

    #plt.title(titles+"--infected node 分布")
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.show()
phase_diagram_infected()