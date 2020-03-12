# demo of Bergman minimal model simulation of insulin modified IVGTT (intravenous glugose tolerance) test

# IVGTT - 2 min 0.3 g/kg glucose and 20 min after that 4 mU/kg/min for 5 min (20 mU/kg over 5 min)

# simulations of normal, obese and t2dm subjects


import numpy as np
import matplotlib.pyplot as plt
from fcns_Bergman import sim_BergmanGI1, sim_BergmanGI2, sim_BergmanGI3
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
Gdata1=np.loadtxt(path+'G_data_fsivgtt_h.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt(path+'I_data_fsivgtt_h.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Gdata1[0,0] = 0
Idata1[0,0] = 0
Gdata1[:,1]=Gdata1[:,1]/18

# obese subjects
Gdata2=np.loadtxt(path+'G_data_fsivgtt_o.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt(path+'I_data_fsivgtt_o.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )
Gdata2[0,0] = 0
Idata2[0,0] = 0
Gdata2[:,1]=Gdata2[:,1]/18

# t2dm subjects
Gdata3=np.loadtxt(path+'G_data_fsivgtt_d.csv', delimiter=',', ndmin=2)
Idata3=np.loadtxt(path+'I_data_fsivgtt_d.csv', delimiter=',', ndmin=2)
Gdata3[:,0] = np.round( Gdata3[:,0] )
Idata3[:,0] = np.round( Idata3[:,0] )
Gdata3[0,0] = 0
Idata3[0,0] = 0
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

# --- glucose subsystem input
g_dose = 0.3 # [g/kg]
glu_bolus = g_dose*1e3/180 # [mmol/kg]
glu_bolus_min = 2 
RaG=np.zeros(idx_final)
RaG[0:int(glu_bolus_min/Ts)]=glu_bolus/Ts/glu_bolus_min

# --- insulin subsystem model input
i_dose = 20 # [mU/kg]
ins_bolus = i_dose # [mU/kg]
ins_bolus_min = 5 
RaI=np.zeros(idx_final)
RaI[int(20/Ts):int(20/Ts)+int(ins_bolus_min/Ts)]=ins_bolus/Ts/ins_bolus_min


# ---- normal subjects
Gb=Gdata1[0,1]
Ib=Idata1[0,1]

parGIh=np.hstack((parG1h[:],parI1h[:]))
x=sim_BergmanGI1(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata1[:,0],Gsim)
Im=resamp(t,Idata1[:,0],Isim)

plt.figure(figsize=(cm2inch(13), cm2inch(24)))
plt.subplot(611)
plt.title(r'blood glucose  (normal subject)')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko')
plt.plot(t,Gsim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Gdata1[:,1],Gm),3)))
plt.subplot(612)
plt.title(r'plasma insulin  (normal subject)')
plt.plot(Idata1[:,0],Idata1[:,1],'ko')
plt.plot(t,Isim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Idata1[:,1],Im),3)))


parGIh=np.hstack((parG1h[:],parI2h[:]))
x=sim_BergmanGI2(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata1[:,0],Gsim)
Im=resamp(t,Idata1[:,0],Isim)
plt.subplot(611)
plt.plot(t,Gsim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Gdata1[:,1],Gm),3)))
plt.subplot(612)
plt.plot(t,Isim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Idata1[:,1],Im),3)))

parGIh=np.hstack((parG1h[:],parI3h[:]))
x=sim_BergmanGI3(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata1[:,0],Gsim)
Im=resamp(t,Idata1[:,0],Isim)
plt.subplot(611)
plt.plot(t,Gsim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Gdata1[:,1],Gm),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(612)
plt.plot(t,Isim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Idata1[:,1],Im),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')


# ---- obese subjects
Gb=Gdata2[0,1]
Ib=Idata2[0,1]

parGIh=np.hstack((parG1o[:],parI1o[:]))
x=sim_BergmanGI1(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata2[:,0],Gsim)
Im=resamp(t,Idata2[:,0],Isim)

plt.subplot(613)
plt.title(r'blood glucose  (obese subject)')
plt.plot(Gdata2[:,0],Gdata2[:,1],'ko')
plt.plot(t,Gsim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Gdata2[:,1],Gm),3)))
plt.subplot(614)
plt.title(r'plasma insulin  (obese subject)')
plt.plot(Idata2[:,0],Idata2[:,1],'ko')
plt.plot(t,Isim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Idata2[:,1],Im),3)))


parGIh=np.hstack((parG1o[:],parI2o[:]))
x=sim_BergmanGI2(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata2[:,0],Gsim)
Im=resamp(t,Idata2[:,0],Isim)
plt.subplot(613)
plt.plot(t,Gsim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Gdata2[:,1],Gm),3)))
plt.subplot(614)
plt.plot(t,Isim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Idata2[:,1],Im),3)))


parGIh=np.hstack((parG1o[:],parI3o[:]))
x=sim_BergmanGI3(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata2[:,0],Gsim)
Im=resamp(t,Idata2[:,0],Isim)
plt.subplot(613)
plt.plot(t,Gsim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Gdata2[:,1],Gm),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(614)
plt.plot(t,Isim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Idata2[:,1],Im),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')


# ---- t2dm subjects
Gb=Gdata3[0,1]
Ib=Idata3[0,1]

parGIh=np.hstack((parG1d[:],parI1d[:]))
x=sim_BergmanGI1(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata3[:,0],Gsim)
Im=resamp(t,Idata3[:,0],Isim)

plt.subplot(615)
plt.title(r'blood glucose  (t2dm subject)')
plt.plot(Gdata3[:,0],Gdata3[:,1],'ko')
plt.plot(t,Gsim,'k',label=r'model A, $R^2$='+str(np.round(r_squared(Gdata3[:,1],Gm),3)))
plt.subplot(616)
plt.title(r'plasma insulin  (t2dm subject)')
plt.plot(Idata3[:,0],Idata3[:,1],'ko')
plt.plot(t,Isim,'k',label='model A, $R^2$='+str(np.round(r_squared(Idata3[:,1],Im),3)))


parGIh=np.hstack((parG1d[:],parI2d[:]))
x=sim_BergmanGI2(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata3[:,0],Gsim)
Im=resamp(t,Idata3[:,0],Isim)
plt.subplot(615)
plt.plot(t,Gsim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Gdata3[:,1],Gm),3)))
plt.subplot(616)
plt.plot(t,Isim,'k--',label=r'model B, $R^2$='+str(np.round(r_squared(Idata3[:,1],Im),3)))


parGIh=np.hstack((parG1d[:],parI3d[:]))
x=sim_BergmanGI3(t,parGIh,RaG,RaI,Gb,Ib)
Gsim=x[:,0]
Isim=x[:,2]
Gm=resamp(t,Gdata3[:,0],Gsim)
Im=resamp(t,Idata3[:,0],Isim)
plt.subplot(615)
plt.plot(t,Gsim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Gdata3[:,1],Gm),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.subplot(616)
plt.plot(t,Isim,'k-.',label=r'model C, $R^2$='+str(np.round(r_squared(Idata3[:,1],Im),3)))
plt.legend()
plt.grid()
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.tight_layout()
plt.savefig('Bergman_demo_ivgtt_output.pdf')
plt.close('all')

