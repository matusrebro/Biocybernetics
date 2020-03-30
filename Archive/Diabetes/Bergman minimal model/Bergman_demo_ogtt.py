# demo of Bergman minimal model simulation of OGTT (oral glugose tolerance test)

# OGTT - 50g of glucose is consumed

# simulations of normal and t2dm subjects

# incretin effect (increase of insulin secretion due to oral glucose intake) is simulated

import numpy as np
import matplotlib.pyplot as plt
from fcns_Bergman import sim_BergmanGIo0, sim_BergmanGIo1, sim_BergmanGIo2, sim_BergmanGIo3
#                           model D         model E         model F            model E

import matplotlib as mpl

def cm2inch(value):
    return value/2.54

mpl.rcParams['figure.autolayout'] = True

###############################################################################

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['xtick.labelsize'] = mpl.rcParams['font.size']

mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['ytick.labelsize'] = mpl.rcParams['font.size']

mpl.rcParams['figure.dpi'] = 119
mpl.rcParams['figure.facecolor'] = 'w'

###############################################################################

mpl.rcParams['font.family'] = [r'Times New Roman']

mpl.rcParams['text.usetex'] = False
#mpl.rcParams['text.usetex'] = True
#mpl.rcParams['text.latex.unicode'] = True
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern}']
#mpl.rcParams['text.latex.preamble'] = [r'\usepackage[T1]{fontenc}', r'\usepackage{txfonts}']

mpl.rcParams['font.size'] = 10
mpl.rcParams['xtick.labelsize'] = 10
mpl.rcParams['ytick.labelsize'] = 10

mpl.rcParams['axes.titlesize'] = mpl.rcParams['font.size']
mpl.rcParams['axes.labelsize'] = mpl.rcParams['font.size']
mpl.rcParams['legend.fontsize'] = mpl.rcParams['font.size']
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.borderpad'] = 0.1
mpl.rcParams['legend.borderaxespad'] = 0.0
mpl.rcParams['legend.labelspacing'] = 0.2

# function for simulation output resampling to match it to measured data
def resamp(t,tn,x):
    ind=np.zeros(len(tn))
    k=0
    for j in range(len(t)):
        if t[j]==tn[k]:
            ind[k]=j
            k+=1    
        if k>=len(tn):
            break
    return x[ind.astype(int)]

# R^2 metric for quantification of model accuracy
def r_squared(y,y_hat):
    y=np.squeeze(y)
    y_hat=np.squeeze(y_hat)
    return 1-np.sum((y-y_hat)**2)/np.sum((y-np.mean(y))**2)

# --- OGTT data import
path='../Data/Vahidi 2013 - IVGTT OGTT/'

BW_healthy = np.squeeze(np.loadtxt(path+'BW_healthy.csv', ndmin=2))
ogtt_gly_healthy = np.loadtxt(path+'OGTT_glycemia_healthy.csv', ndmin=2)
ogtt_ins_healthy = np.loadtxt(path+'OGTT_insulin_healthy.csv', ndmin=2)

BW_t2dm = np.squeeze(np.loadtxt(path+'BW_t2dm.csv', ndmin=2))
ogtt_gly_t2dm = np.loadtxt(path+'OGTT_glycemia_t2dm.csv', ndmin=2)
ogtt_ins_t2dm = np.loadtxt(path+'OGTT_insulin_t2dm.csv', ndmin=2)

BW_healthy_mean=np.mean(BW_healthy)
ogtt_gly_healthy_mean=np.mean(ogtt_gly_healthy[:,1:],1)
ogtt_ins_healthy_mean=np.mean(ogtt_ins_healthy[:,1:],1)

BW_t2dm_mean=np.mean(BW_t2dm)
ogtt_gly_t2dm_mean=np.mean(ogtt_gly_t2dm[:,1:],1)
ogtt_ins_t2dm_mean=np.mean(ogtt_ins_t2dm[:,1:],1)

Ts=1
t=np.arange(0,ogtt_gly_t2dm[-1,0]+Ts,Ts)
tsim=ogtt_gly_t2dm[-1,0]
idx_final=t.size

# --- normal subjects
d=np.zeros_like(t)
d[0]=50*1e3/180/Ts/BW_healthy_mean

Gb=ogtt_gly_healthy_mean[0]
Ib=ogtt_ins_healthy_mean[0]/6

parG=np.hstack((np.loadtxt('parG1h3.csv'),np.loadtxt('parG1h3o.csv')))
parI2h=np.loadtxt('parI2h3.csv')[0:4]
parI=np.hstack((np.loadtxt('parI2h3.csv')[0:4],np.loadtxt('parI2h3o0.csv')))
par=np.hstack((parG, parI))
#Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, Kg1m, Kg2m          

x=sim_BergmanGIo0(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_healthy[:,0],Gsim)
Im=resamp(t,ogtt_ins_healthy[:,0],Isim)


plt.figure(figsize=(cm2inch(13), cm2inch(10)))
plt.subplot(211)
plt.title(r'plasma glucose')
plt.plot(ogtt_gly_healthy[:,0],ogtt_gly_healthy_mean,'ko')
plt.plot(t,Gsim,'k',label=r'model D, $R^2$='+str(np.round(r_squared(ogtt_gly_healthy_mean,Gm),3)))
plt.subplot(212)
plt.title(r'plasma insulin')
plt.plot(ogtt_ins_healthy[:,0],ogtt_ins_healthy_mean/6,'ko')
plt.plot(t,Isim,'k',label=r'model D, $R^2$='+str(np.round(r_squared(ogtt_ins_healthy_mean/6,Im),3)))

parI=np.hstack((np.loadtxt('parI2h3.csv')[0:4],np.loadtxt('parI2h3o1.csv')))
par=np.hstack((parG, parI))

#Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, Kg3a, Kg3b
x=sim_BergmanGIo1(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_healthy[:,0],Gsim)
Im=resamp(t,ogtt_ins_healthy[:,0],Isim)
plt.subplot(211)
plt.plot(t,Gsim,'k--',label=r'model E, $R^2$='+str(np.round(r_squared(ogtt_gly_healthy_mean,Gm),3)))
plt.subplot(212)
plt.plot(t,Isim,'k--',label=r'model E, $R^2$='+str(np.round(r_squared(ogtt_ins_healthy_mean/6,Im),3)))

parI=np.hstack((np.loadtxt('parI2h3.csv')[0:4],np.loadtxt('parI2h3o2.csv')))
par=np.hstack((parG, parI))

#Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, T3, Kg3a, Kg3b
x=sim_BergmanGIo2(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_healthy[:,0],Gsim)
Im=resamp(t,ogtt_ins_healthy[:,0],Isim)
plt.subplot(211)
plt.plot(t,Gsim,'k-.',label=r'model F, $R^2$='+str(np.round(r_squared(ogtt_gly_healthy_mean,Gm),3)))
plt.subplot(212)
plt.plot(t,Isim,'k-.',label=r'model F, $R^2$='+str(np.round(r_squared(ogtt_ins_healthy_mean/6,Im),3)))

parI=np.hstack((np.loadtxt('parI2h3.csv')[0:4],np.loadtxt('parI2h3o3.csv')))
par=np.hstack((parG, parI))

# Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, Kg1m, Kg2m, T3, Kg3a, Kg3b
x=sim_BergmanGIo3(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_healthy[:,0],Gsim)
Im=resamp(t,ogtt_ins_healthy[:,0],Isim)
plt.subplot(211)
plt.plot(t,Gsim,'k:',label=r'model G, $R^2$='+str(np.round(r_squared(ogtt_gly_healthy_mean,Gm),3)))
plt.grid()
plt.legend()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(212)
plt.plot(t,Isim,'k:',label=r'model G, $R^2$='+str(np.round(r_squared(ogtt_ins_healthy_mean/6,Im),3)))
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.legend()
plt.tight_layout()
plt.savefig('Bergman_demo_OGTT_normal_output.pdf')
plt.close('all')

# --- t2dm subjects
d=np.zeros_like(t)
d[0]=50*1e3/180/Ts/BW_t2dm_mean

Gb=ogtt_gly_t2dm_mean[0]
Ib=ogtt_ins_t2dm_mean[0]/6

parG=np.hstack((np.loadtxt('parG1d3.csv'),np.loadtxt('parG1d3o.csv')))
parI2d=np.loadtxt('parI2d3.csv')[0:4]
parI=np.hstack((np.loadtxt('parI2d3.csv')[0:4],np.loadtxt('parI2d3o0.csv')))
par=np.hstack((parG, parI))

#Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, Kg1m, Kg2m  
x=sim_BergmanGIo0(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_t2dm[:,0],Gsim)
Im=resamp(t,ogtt_ins_t2dm[:,0],Isim)

plt.figure(figsize=(cm2inch(13), cm2inch(10)))
plt.subplot(211)
plt.title(r'plasma glucose')
plt.plot(ogtt_gly_t2dm[:,0],ogtt_gly_t2dm_mean,'ko')
plt.plot(t,Gsim,'k',label=r'model D, $R^2$='+str(np.round(r_squared(ogtt_gly_t2dm_mean,Gm),3)))
plt.subplot(212)
plt.title(r'plasma insulin')
plt.plot(ogtt_ins_t2dm[:,0],ogtt_ins_t2dm_mean/6,'ko')
plt.plot(t,Isim,'k',label=r'model D, $R^2$='+str(np.round(r_squared(ogtt_ins_t2dm_mean/6,Im),3)))

parI=np.hstack((np.loadtxt('parI2d3.csv')[0:4],np.loadtxt('parI2d3o1.csv')))
par=np.hstack((parG, parI))

x=sim_BergmanGIo1(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_t2dm[:,0],Gsim)
Im=resamp(t,ogtt_ins_t2dm[:,0],Isim)

plt.subplot(211)
plt.plot(t,Gsim,'k--',label=r'model E, $R^2$='+str(np.round(r_squared(ogtt_gly_t2dm_mean,Gm),3)))
plt.subplot(212)
plt.plot(t,Isim,'k--',label=r'model E, $R^2$='+str(np.round(r_squared(ogtt_ins_t2dm_mean/6,Im),3)))

parI=np.hstack((np.loadtxt('parI2d3.csv')[0:4],np.loadtxt('parI2d3o2.csv')))
par=np.hstack((parG, parI))

x=sim_BergmanGIo2(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_t2dm[:,0],Gsim)
Im=resamp(t,ogtt_ins_t2dm[:,0],Isim)

plt.subplot(211)
plt.plot(t,Gsim,'k-.',label=r'model F, $R^2$='+str(np.round(r_squared(ogtt_gly_t2dm_mean,Gm),3)))
plt.subplot(212)
plt.plot(t,Isim,'k-.',label=r'model F, $R^2$='+str(np.round(r_squared(ogtt_ins_t2dm_mean/6,Im),3)))

parI=np.hstack((np.loadtxt('parI2d3.csv')[0:4],np.loadtxt('parI2d3o3.csv')))
par=np.hstack((parG, parI))

x=sim_BergmanGIo3(t,par,d,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,4]
Gm=resamp(t,ogtt_gly_t2dm[:,0],Gsim)
Im=resamp(t,ogtt_ins_t2dm[:,0],Isim)

plt.subplot(211)
plt.plot(t,Gsim,'k:',label=r'model G, $R^2$='+str(np.round(r_squared(ogtt_gly_t2dm_mean,Gm),3)))
plt.grid()
plt.legend()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(212)
plt.plot(t,Isim,'k:',label=r'model G, $R^2$='+str(np.round(r_squared(ogtt_ins_t2dm_mean/6,Im),3)))
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.legend()
plt.tight_layout()
plt.savefig('Bergman_demo_OGTT_t2dm_output.pdf')
plt.close('all')
