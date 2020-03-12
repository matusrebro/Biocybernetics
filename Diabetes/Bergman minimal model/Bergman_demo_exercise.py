# demo of Bergman minimal model simulation 
# in combination with model of exercise effect on glucose and insulin levels

# simulations of normal subject
# subject exercised at theoretical 40 % of VO2max for 60 minutes

import numpy as np
import matplotlib.pyplot as plt
from fcns_Bergman import sim_Ex

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

# data import
path='../Data/Roy 2007 - Exercise effects/'

# normal subjects
Gdata=np.loadtxt(path+'Gexdata.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt(path+'Iexdata.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )
Gdata[:,1]=Gdata[:,1]/18

# parameters of insulin-glucose system for normal subject
parG1h=np.loadtxt('parG1h.csv')
parI2h=np.loadtxt('parI2h.csv')

# parameters of subsystems of dynamics of glucose uptake and production during exercise
parGprodb=np.loadtxt('parGprodb.csv')
parGupb=np.loadtxt('parGupb.csv')

a1, a2 = parGprodb
a3, a4 = parGupb
a1=a1/180 # conversion to mmol
a3=a3/180 # conversion to mmol

# remaining parameters
a5, a6, k, Tgly = np.loadtxt('parGgly.csv')

# getting all parameters together...
parGIh=np.hstack((parG1h[:],parI2h[:]))
Tg, Kx, V_G, Tx, Ti, Kg1, Kg2, T2, V_I  = parGIh    
a5, a6, k, Tgly = np.loadtxt('parGgly.csv')
p=[Tg, Kx, V_G, Tx, Ti, Kg1, Kg2, T2, V_I, a1, a2, a3, a4, a5, a6, k, Tgly]

# basal state from data
Gb=Gdata[0,1]
Ib=Idata[0,1]

# simulation and data collection
tt=np.arange(0,125,1)
Ts=tt[1]-tt[0]
ux=np.zeros_like(tt)
ux[0:int(60/Ts)]=40
x=sim_Ex(tt,p,ux,Gb,Ib)
Gsim=x[:,0]
Gsimm=resamp(tt,Gdata[:,0],Gsim)
Isim=x[:,2]
Isimm=resamp(tt,Idata[:,0],Isim)

plt.figure()
plt.subplot(211)
plt.title(r'Plasma glucose, $R^2$='+str(np.round(r_squared(Gdata[:,1],Gsimm),3)))
plt.plot(Gdata[:,0],Gdata[:,1],'ko')
plt.plot(tt,Gsim,'k')
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.grid()
plt.subplot(212)
plt.title(r'Plasma insulin, $R^2$='+str(np.round(r_squared(Idata[:,1],Isimm),3)))
plt.plot(Idata[:,0],Idata[:,1],'ko')
plt.plot(tt,Isim,'k')
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.grid()
plt.tight_layout()
plt.savefig('Bergman_demo_exercise_output.pdf')
plt.close('all')

