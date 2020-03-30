# demo of Bergman minimal model simulation of hyperinsulinemic euglycemic glucose clamp

# Clamp - constant 120 mU/m2/min insulin infusion
#         glucose infusion controlled by closed-loop algorithm

# simulations of normal, obese and t2dm subjects


import numpy as np
import matplotlib.pyplot as plt
from fcns_Bergman import sim_BergmanGI1_clamp, sim_BergmanGI2_clamp, sim_BergmanGI3_clamp
#                           model A         model B         model C

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


# IVGTT data import
path='../Data/Katz 2000 IVGTT Clamp/'

# normal subjects
Gdata1=np.loadtxt(path+'G_data_Gclamp_h.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt(path+'I_data_Gclamp_h.csv', delimiter=',', ndmin=2)
Radata1=np.loadtxt(path+'Ra_data_Gclamp_h.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Radata1[:,0] = np.round( Radata1[:,0] )
Gdata1[0,0] = 0
Idata1[0,0] = 0
Radata1[0,0] = 0
Gdata1[:,1]=Gdata1[:,1]/18

# obese subjects
Gdata2=np.loadtxt(path+'G_data_Gclamp_o.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt(path+'I_data_Gclamp_o.csv', delimiter=',', ndmin=2)
Radata2=np.loadtxt(path+'Ra_data_Gclamp_o.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )
Radata2[:,0] = np.round( Radata2[:,0] )
Gdata2[0,0] = 0
Idata2[0,0] = 0
Radata2[0,0] = 0
Gdata2[:,1]=Gdata2[:,1]/18

# t2dm subjects
Gdata3=np.loadtxt(path+'G_data_Gclamp_d.csv', delimiter=',', ndmin=2)
Idata3=np.loadtxt(path+'I_data_Gclamp_d.csv', delimiter=',', ndmin=2)
Radata3=np.loadtxt(path+'Ra_data_Gclamp_d.csv', delimiter=',', ndmin=2)
Gdata3[:,0] = np.round( Gdata3[:,0] )
Idata3[:,0] = np.round( Idata3[:,0] )
Radata3[:,0] = np.round( Radata3[:,0] )
Gdata3[0,0] = 0
Idata3[0,0] = 0
Radata3[0,0] = 0
Gdata3[:,1]=Gdata3[:,1]/18


# --- glucose subsystem model parameters
parG1h=np.loadtxt('parG1h.csv') # normal
parG1o=np.loadtxt('parG1o.csv') # obese
parG1d=np.loadtxt('parG1d.csv') # t2dm


# --- insulin subsystem model parameters

#  first model (P control structure - model A) normal
parI1h=np.loadtxt('parI1h.csv')
# Ti, Kg1, V_I
#  second model (PD control structure - model B) normal
parI2h=np.loadtxt('parI2h.csv')
# Ti, Kg1, Kg2, T2, V_I
#  third model (PD control structure 2 - model C) normal
parI3h=np.loadtxt('parI3h.csv')
# Ti, Kg1, T1, Kg2, T2, V_I

#  first model (P control structure - model A) obese
parI1o=np.loadtxt('parI1o.csv')
#  second model (PD control structure - model B) obese
parI2o=np.loadtxt('parI2o.csv')
#  third model (PD control structure 2 - model C) obese
parI3o=np.loadtxt('parI3o.csv')

#  first model (P control structure - model A) t2dm
parI1d=np.loadtxt('parI1d.csv')
#  second model (PD control structure - model B) t2dm
parI2d=np.loadtxt('parI2d.csv')
#  third model (PD control structure 2 - model C) t2dm
parI3d=np.loadtxt('parI3d.csv')


Ts=1
tsim=Gdata1[-1,0]
t=np.arange(Gdata1[0,0],tsim+Ts,Ts)
idx_final=t.size

# ---- normal subjects
BW=70 # we assume that the average normal subject's bodyweight is 70kg
Gb=Gdata1[0,1]
Ib=Idata1[0,1]

# - insulin subsystem model input
RaI=np.zeros(idx_final)
RaI[:]=120*1.9/BW # here we assume that the average normal subject's body surface area is 1.9 m^2

# --- clamp

parGIh=np.hstack((parG1h[:],parI1h[:]))
x, RaG2=sim_BergmanGI1_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata1[:,0],Gsim)
Im=resamp(t,Idata1[:,0],Isim)


plt.figure(figsize=(cm2inch(13), cm2inch(15)))
plt.subplot(311)
plt.title(r'blood glucose')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko')
plt.plot(t,Gsim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Gdata1[:,1],Gm),3)))
plt.subplot(312)
plt.title(r'plasma insulin')
plt.plot(Idata1[:,0],Idata1[:,1],'ko')
plt.plot(t,Isim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Idata1[:,1],Im),3)))
plt.subplot(313)
plt.title(r'glucose infusion')
plt.plot(Radata1[:,0],Radata1[:,1]/180/BW,'ko')
plt.plot(t,RaG2,'k',label=r'model A')


parGIh=np.hstack((parG1h[:],parI2h[:]))
x, RaG2=sim_BergmanGI2_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata1[:,0],Gsim)
Im=resamp(t,Idata1[:,0],Isim)
plt.subplot(311)
plt.plot(t,Gsim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Gdata1[:,1],Gm),3)))
plt.subplot(312)
plt.plot(t,Isim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Idata1[:,1],Im),3)))
plt.subplot(313)
plt.plot(t,RaG2,'k--',label=r'model B')

parGIh=np.hstack((parG1h[:],parI3h[:]))
x, RaG2=sim_BergmanGI3_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata1[:,0],Gsim)
Im=resamp(t,Idata1[:,0],Isim)
plt.subplot(311)
plt.plot(t,Gsim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Gdata1[:,1],Gm),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(312)
plt.plot(t,Isim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Idata1[:,1],Im),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.subplot(313)
plt.plot(t,RaG2,'k-.',label=r'model C')
plt.legend(loc=(0.74,0.4))
plt.grid()
plt.ylabel(r'$Ra_G(t)$ [mmol/kg/min]')
plt.xlabel(r'time [min]')
plt.tight_layout()
plt.savefig('Bergman_demo_eugc_normal_output.pdf')
plt.close('all')

# ---- obese subjects
BW=100 # we assume that the average normal subject's bodyweight is 100kg
Gb=Gdata2[0,1]
Ib=Idata2[0,1]

# - insulin subsystem model input
RaI=np.zeros(idx_final)
RaI[:]=120*2.2/BW # here we assume that the average normal subject's body surface area is 2.2 m^2

parGIh=np.hstack((parG1o[:],parI1o[:]))
x, RaG2=sim_BergmanGI1_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata2[:,0],Gsim)
Im=resamp(t,Idata2[:,0],Isim)


plt.figure(figsize=(cm2inch(13), cm2inch(15)))
plt.subplot(311)
plt.title(r'blood glucose')
plt.plot(Gdata2[:,0],Gdata2[:,1],'ko')
plt.plot(t,Gsim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Gdata2[:,1],Gm),3)))
plt.subplot(312)
plt.title(r'plasma insulin')
plt.plot(Idata2[:,0],Idata2[:,1],'ko')
plt.plot(t,Isim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Idata2[:,1],Im),3)))
plt.subplot(313)
plt.title(r'glucose infusion')
plt.plot(Radata2[:,0],Radata2[:,1]/180/BW,'ko')
plt.plot(t,RaG2,'k',label=r'model A')


parGIh=np.hstack((parG1o[:],parI2o[:]))
x, RaG2=sim_BergmanGI2_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata2[:,0],Gsim)
Im=resamp(t,Idata2[:,0],Isim)
plt.subplot(311)
plt.plot(t,Gsim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Gdata2[:,1],Gm),3)))
plt.subplot(312)
plt.plot(t,Isim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Idata2[:,1],Im),3)))
plt.subplot(313)
plt.plot(t,RaG2,'k--',label=r'model B')

parGIh=np.hstack((parG1o[:],parI3o[:]))
x, RaG2=sim_BergmanGI3_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata2[:,0],Gsim)
Im=resamp(t,Idata2[:,0],Isim)
plt.subplot(311)
plt.plot(t,Gsim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Gdata2[:,1],Gm),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(312)
plt.plot(t,Isim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Idata2[:,1],Im),3)))
plt.legend(loc=(0.6,0.35))
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.subplot(313)
plt.plot(t,RaG2,'k-.',label=r'model C')
plt.legend(loc=(0.74,0.35))
plt.grid()
plt.ylabel(r'$Ra_G(t)$ [mmol/kg/min]')
plt.xlabel(r'time [min]')
plt.tight_layout()
plt.savefig('Bergman_demo_eugc_obese_output.pdf')
plt.close('all')

# ---- t2dm subjects
BW=100
Gb=Gdata3[0,1]
Ib=Idata3[0,1]

# - insulin subsystem model input
RaI=np.zeros(idx_final)
RaI[:]=120*2.2/BW

# --- clamp
parGIh=np.hstack((parG1d[:],parI1d[:]))
x, RaG2=sim_BergmanGI1_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata3[:,0],Gsim)
Im=resamp(t,Idata3[:,0],Isim)

plt.figure(figsize=(cm2inch(13), cm2inch(15)))
plt.subplot(311)
plt.title(r'blood glucose')
plt.plot(Gdata3[:,0],Gdata3[:,1],'ko')
plt.plot(t,Gsim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Gdata3[:,1],Gm),3)))
plt.subplot(312)
plt.title(r'plasma insulin')
plt.plot(Idata3[:,0],Idata3[:,1],'ko')
plt.plot(t,Isim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Idata3[:,1],Im),3)))
plt.subplot(313)
plt.title(r'glucose infusion')
plt.plot(Radata3[:,0],Radata3[:,1]/180/BW,'ko')
plt.plot(t,RaG2,'k',label=r'model A')

parGIh=np.hstack((parG1d[:],parI2d[:]))
x, RaG2=sim_BergmanGI2_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata3[:,0],Gsim)
Im=resamp(t,Idata3[:,0],Isim)
plt.subplot(311)
plt.plot(t,Gsim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Gdata3[:,1],Gm),3)))
plt.subplot(312)
plt.plot(t,Isim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Idata3[:,1],Im),3)))
plt.subplot(313)
plt.plot(t,RaG2,'k--',label=r'model B')

parGIh=np.hstack((parG1d[:],parI3d[:]))
x, RaG2=sim_BergmanGI3_clamp(t,parGIh,RaI,Gb,Ib,BW)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata3[:,0],Gsim)
Im=resamp(t,Idata3[:,0],Isim)
plt.subplot(311)
plt.plot(t,Gsim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Gdata3[:,1],Gm),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(312)
plt.plot(t,Isim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Idata3[:,1],Im),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.subplot(313)
plt.plot(t,RaG2,'k-.',label=r'model C')
plt.legend(loc=(0.74,0.4))
plt.grid()
plt.ylabel(r'$Ra_G(t)$ [mmol/kg/min]')
plt.xlabel(r'time [min]')
plt.tight_layout()
plt.savefig('Bergman_demo_eugc_t2dm_output.pdf')
plt.close('all')
