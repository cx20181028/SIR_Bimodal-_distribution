import os

import netCDF4 as nc
import xarray  as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 约定俗成的写法plt

# 安装cfgrib需先安装二进制共享库
# import cfgrib
'''
温度的单位是k,开尔文单位，T=t+273.15
-aco2nee --累积二氧化碳净生态系统交换	公斤米-2
生态系统CO2净交换(net ecosystem exchange,缩写为NEE)指生态系统与大气圈之间净的CO2交换
t2m--2m 温度
variables(dimensions): float32 longitude(longitude), 
float32 latitude(latitude),
 int32 time(time), 
int16 t2m(time, latitude, longitude), int16 aco2nee(time, latitude, longitude)
'''

file = ["CAMS_total.nc", 'CAMS_left_top.nc', "CAMS_left_down.nc", "CAMS _right_top.nc", "CAMS_right_down.nc"]


def readData(file):


    dataset = nc.Dataset(file, 'r+')
    t = dataset.variables.keys()
    print(dataset.variables.keys())
    longitude = dataset.variables['longitude']
    latitude = dataset.variables['latitude']
    t2m = dataset.variables['t2m']
    co2 = dataset.variables['aco2nee']
    time = dataset.variables['time']
    co2 = np.array(co2[:])
    print("co2的shape %s" % (str(co2.shape)))
    print(co2[1, :, :])
    longitude = np.array(longitude[:])
    print("longitude的shape %s" % (str(longitude.shape)))
    latitude = np.array(latitude[:])
    print("latitude的shape %s" % (str(latitude.shape)))
    t2m = np.array(t2m[:])
    print("t2m的shape %s" % (str(t2m.shape)))
    # print(t2m[:])
    time = np.array(time[:])
    print(time.shape)
    # print(pd.to_datetime(time).year)
    # print(time[:])
    dataset = nc.Dataset("levtype_sfc.nc", 'r+')
    t = dataset.variables.keys()
    print(dataset.variables.keys())
    datasets = nc.Dataset("levtype_pl.nc", 'r+')
    longitude = dataset.variables['longitude']
    latitude = dataset.variables['latitude']
    t2m = dataset.variables['t2m']
    co2 = datasets.variables['co2']
    time = dataset.variables['time']
    return co2, t2m


def paint_month(array1, array2,mark,filename):
    X = np.arange(1, 13, 1)
    year=np.arange(2003,2021,1)
    co2 = array1
    t2m = array2
    length = array1.shape[0]
    index = 0
    t2m_y = []
    co2_y = []
    epoch = 0
    markname = ''
    while index < length:
        co2mid = co2[index, :, :]
        t2mmid = t2m[index, :, :]
        if mark == "max":
            co2_y.append(np.max(co2mid) * 10 * 10 * 10)
            t2m_y.append(np.max(t2mmid) - 273.15)
            markname = "最大"
        elif mark == "mean":
            co2_y.append(np.mean(co2mid) * 10 * 10 * 10)
            t2m_y.append(np.mean(t2mmid) - 273.15)
            markname = '平均'
        else:
            co2_y.append(np.min(co2mid) * 10 * 10 * 10)
            t2m_y.append(np.min(t2mmid) - 273.15)
            markname = '最小'


        if (index + 1) % 12 == 0:
            print("第%dci" % (epoch))

            t2ms = np.array(t2m_y)
            co2s = np.array(co2_y)
            plt.plot(X, co2s, label="CO2")
            plt.plot(X, t2ms, label="Temperature")
            plt.xlabel("Month")
            plt.legend(loc="upper left")
            plt.title(str(year[epoch])+"温度和co2净生态系统交换量(x1000倍)"+markname+"值"+filename)
            # 设置纵轴标签
            plt.ylabel("CO2/Temperature")
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            plt.rcParams['axes.unicode_minus'] = False
            # 显示中文标签 plt.rcParams[‘axes.unicode_minus’]=False
            # 在ipython的交互环境中需要这句话才能显示出来
            pre_savename = "./image_total/image_paint_each-month1/"+mark+"/"
            fname = str(epoch) +filename+ '温度和co2净交换量(x1000倍)'+markname+'值.png'
            savename = os.path.join(pre_savename, fname)
            plt.savefig(savename, dpi=900)
            plt.show()
            epoch = epoch + 1
            t2m_y = []
            co2_y = []
        index = index + 1


def paint_mean(array1, array2):
    X = np.arange(1, 13, 1)
    co2 = array1
    t2m = array2
    length = array1.shape[0]
    index = 0
    t2m_y = []
    co2_y = []
    epoch = 0
    while index < length:
        co2mid = co2[index, :, :]
        t2mmid = t2m[index, :, :]
        co2_y.append(np.mean(co2mid) * 10 * 10 * 10)
        t2m_y.append(np.mean(t2mmid) - 273.15)

        if (index + 1) % 12 == 0:
            print("第%dci" % (epoch))

            t2ms = np.array(t2m_y)
            co2s = np.array(co2_y)
            plt.plot(X, co2s, label="CO2")
            plt.plot(X, t2ms, label="Temperature")
            plt.xlabel("Month")
            plt.legend(loc="upper left")
            plt.title("温度和co2净生态系统交换量(x1000倍)平均值")
            # 设置纵轴标签
            plt.ylabel("CO2/Temperature")
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            plt.rcParams['axes.unicode_minus'] = False
            # 显示中文标签 plt.rcParams[‘axes.unicode_minus’]=False
            # 在ipython的交互环境中需要这句话才能显示出来
            pre_savename = "./image_total/image_paint_each-month/mean/"
            fname = str(epoch) + '温度和co2净交换量(x1000倍)平均值.png'
            savename = os.path.join(pre_savename, fname)
            plt.savefig(savename, dpi=900)
            plt.show()
            epoch = epoch + 1
            t2m_y = []
            co2_y = []
        index = index + 1


def paint_min(array1, array2):
    X = np.arange(1, 13, 1)
    co2 = array1
    t2m = array2
    length = array1.shape[0]
    index = 0
    t2m_y = []
    co2_y = []
    epoch = 0
    while index < length:
        co2mid = co2[index, :, :]
        t2mmid = t2m[index, :, :]
        co2_y.append(np.min(co2mid) * 10 * 10 * 10)
        t2m_y.append(np.min(t2mmid) - 273.15)

        if (index + 1) % 12 == 0:
            print("第%dci" % (epoch))

            t2ms = np.array(t2m_y)
            co2s = np.array(co2_y)
            plt.plot(X, co2s, label="CO2")
            plt.plot(X, t2ms, label="Temperature")
            plt.xlabel("Month")
            plt.legend(loc="upper left")
            plt.title("温度和co2净生态系统交换量(x1000倍)最小值")
            # 设置纵轴标签
            plt.ylabel("CO2/Temperature")
            plt.xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
            plt.rcParams['axes.unicode_minus'] = False
            # 显示中文标签 plt.rcParams[‘axes.unicode_minus’]=False
            # 在ipython的交互环境中需要这句话才能显示出来
            pre_savename = "./image_total/image_paint_each-month/min/"
            fname = str(epoch) + '温度和co2净交换量(x1000倍)最小值.png'
            savename = os.path.join(pre_savename, fname)
            #plt.savefig(savename, dpi=900)
            plt.show()
            epoch = epoch + 1
            t2m_y = []
            co2_y = []
        index = index + 1
def paint_year(array1, array2,mark,filename):
    X = np.arange(2003,2021,1)
    co2 = array1
    t2m = array2
    length = array1.shape[0]
    index = 0
    t2m_y = []
    co2_y = []
    t2m_yy = []
    co2_yy = []
    markname=''
    while index < length:
        co2mid = co2[index, :, :]
        t2mmid = t2m[index, :, :]
        co2_y.append(co2mid)
        t2m_y.append(t2mmid)

        if (index + 1) % 12 == 0:
            t2ms = np.array(t2m_y)
            co2s = np.array(co2_y)
            if mark=="max":
                co2_yy.append(np.max(co2s) * 10 * 10 * 10*10)
                t2m_yy.append(np.max(t2ms)- 273.15)
                markname="最大"
            elif mark=="mean":
                co2_yy.append(np.mean(co2s) * 10 * 10 * 10*10)
                t2m_yy.append(np.mean(t2ms)- 273.15)
                markname='均值'
            else:
                co2_yy.append(np.min(co2s) * 10 * 10 * 10)
                t2m_yy.append(np.min(t2ms)- 273.15)
                markname='最小'
            t2m_y = []
            co2_y = []
        index = index + 1
    plt.plot(X, co2_yy, label="CO2")
    plt.plot(X, t2m_yy, label="Temperature")
    plt.xlabel("Year")
    plt.legend(loc="center left")
    plt.title("温度和co2净生态系统交换量(x1000倍)"+markname+"值"+filename)
    # 设置纵轴标签
    plt.ylabel("CO2/Temperature")
    plt.xticks([2003, 2005,2007,2009,2011,2013,2015,2017,2019,2020])
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    # 显示中文标签 plt.rcParams[‘axes.unicode_minus’]=False
    # 在ipython的交互环境中需要这句话才能显示出来
    pre_savename = "./image_total/image_paint_each-year/"
    fname = filename+'温度和co2净交换量(x1000倍)'+markname+'值.png'
    savename = os.path.join(pre_savename, fname)
    plt.savefig(savename, dpi=900)
    plt.show()
if __name__ == '__main__':
   co2, t2m = readData(file[0])
  # paint_min(co2,t2m)
  # paint_year(co2, t2m,"mean",file[0])
'''



    for fname in file:
        co2, t2m = readData(fname)
        mark=["max","mean","min"]
        for i in  mark:
             #paint_year(co2, t2m,i,fname)
             paint_month(co2,t2m,i,fname)
'''