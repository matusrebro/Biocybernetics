# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 10:21:59 2018

@author: Admin
"""

# Data published in Dissertation:
# A physiologic model of glucose metabolism in man and its use to design and assess improved insulin therapies for diabetes
# by Sorensen 1978
# all data were extracted from graphs
# contains data from various experiments including:
# glucose and insulin tolerance tests, clamp and continous infusion studies
# all data are of normal (healthy) subjects

# blood glucose conc. = 84 % whole blood water glucose conc.
# plasma glucose conc. = 92.5 % whole blood water glucose conc.

# mean subject bodyweight is 70 kg for all data

# unit dimensions:

# G [mg/dl]
# I [mU/L]
# glucose infusion [g]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


# --- plot settings

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

#mpl.rcParams['text.usetex'] = False
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.unicode'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage[T1]{fontenc}', r'\usepackage{lmodern}']
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
mpl.rcParams['legend.numpoints'] = 1


def cm2inch(value):
    return value/2.54


fonts=8

# --- IVGTT dataset_01
# first data set is from 0.5 g/kg over 3 min IVGTT
# presented data are mean of 110 normal adult males 
# blood glucose and plasma insulin data
# page 271


Gdata=np.loadtxt('G_data.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt('I_data.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )


plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT (3 min 0.5g/kg) data of normal subjects')
plt.plot(Gdata[:,0],Gdata[:,1],'ko-')
plt.ylabel(r'blood glucose [mg/dL]',fontsize=fonts)
plt.subplot(212)
plt.plot(Idata[:,0],Idata[:,1],'ko-')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ivgtt_healthy_01.pdf')
plt.close('all')

# --- IVGTT dataset_02

# mean data for various glucose loads of 15 normal subjects
# glucose is infused over 3 minutes
# blood glucose and plasma insulin data
# page 274

# 0.05 g/kg    
Gdata05=np.loadtxt('G_data_ivgtt_05.csv', delimiter=',', ndmin=2)
Idata05=np.loadtxt('I_data_ivgtt_05.csv', delimiter=',', ndmin=2)
Gdata05[:,0] = np.round( Gdata05[:,0] )
Idata05[:,0] = np.round( Idata05[:,0] )
# 0.20 g/kg    
Gdata20=np.loadtxt('G_data_ivgtt_20.csv', delimiter=',', ndmin=2)
Idata20=np.loadtxt('I_data_ivgtt_20.csv', delimiter=',', ndmin=2)
Gdata20[:,0] = np.round( Gdata20[:,0] )
Idata20[:,0] = np.round( Idata20[:,0] )
# 0.50 g/kg    
Gdata50=np.loadtxt('G_data_ivgtt_50.csv', delimiter=',', ndmin=2)
Idata50=np.loadtxt('I_data_ivgtt_50.csv', delimiter=',', ndmin=2)
Gdata50[:,0] = np.round( Gdata50[:,0] )
Idata50[:,0] = np.round( Idata50[:,0] )
# 0.75 g/kg     
Gdata75=np.loadtxt('G_data_ivgtt_75.csv', delimiter=',', ndmin=2)
Idata75=np.loadtxt('I_data_ivgtt_75.csv', delimiter=',', ndmin=2)
Gdata75[:,0] = np.round( Gdata75[:,0] )
Idata75[:,0] = np.round( Idata75[:,0] )

plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT data of normal subjects')
plt.plot(Gdata05[:,0],Gdata05[:,1],'ko-',label='0.05 g/kg')
plt.plot(Gdata20[:,0],Gdata20[:,1],'kx-',label='0.2 g/kg')
plt.plot(Gdata50[:,0],Gdata50[:,1],'ks-',label='0.5 g/kg')
plt.plot(Gdata75[:,0],Gdata75[:,1],'k*-',label='0.75 g/kg')
plt.ylabel(r'blood glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata05[:,0],Idata05[:,1],'ko-',label='0.05 g/kg')
plt.plot(Idata20[:,0],Idata20[:,1],'kx-',label='0.2 g/kg')
plt.plot(Idata50[:,0],Idata50[:,1],'ks-',label='0.5 g/kg')
plt.plot(Idata75[:,0],Idata75[:,1],'k*-',label='0.75 g/kg')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ivgtt_healthy_02.pdf')
plt.close('all')


# --- IVITT data

# 0.04 U/kg Intravenous Insulin Tolerance Test
# mean data of 15 normal subjects
# insulin is infused over 3 minutes
# plasma glucose and plasma glucagon data (no insulin data, Ib=13mU/L is assumed)
# page 277


Gdata=np.loadtxt('G_data_ivitt_04.csv', delimiter=',', ndmin=2)
Gadata=np.loadtxt('Ga_data_ivitt_04.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Gadata[:,0] = np.round( Gadata[:,0] )

plt.figure()
plt.subplot(211)
plt.title(r'mean IVITT data of normal subjects')
plt.plot(Gdata[:,0],Gdata[:,1],'ko-')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Gadata[:,0],Gadata[:,1],'ko-')
plt.ylabel(r'plasma glucagon [pg/mL]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ivitt_healthy.pdf')
plt.close('all')


# --- CII data
# CII - continuous insulin infusion

# 0.25 mU/kg/min  - mean data of 5 normal subjects
# 0.4 mU/kg/min  - mean data of 6 normal subjects
# plasma glucose and plasma insulin data for first case
# plasma glucose, plasma insulin and plasma glucagon data for second case
# page 279

Gdata25=np.loadtxt('G_data_cii_25.csv', delimiter=',', ndmin=2)
Idata25=np.loadtxt('I_data_cii_25.csv', delimiter=',', ndmin=2)
Gdata25[:,0] = np.round( Gdata25[:,0] )
Idata25[:,0] = np.round( Idata25[:,0] )

Gdata40=np.loadtxt('G_data_cii_40.csv', delimiter=',', ndmin=2)
Idata40=np.loadtxt('I_data_cii_40.csv', delimiter=',', ndmin=2)
Gadata40=np.loadtxt('Ga_data_cii_40.csv', delimiter=',', ndmin=2)
Gdata40[:,0] = np.round( Gdata40[:,0] )
Idata40[:,0] = np.round( Idata40[:,0] )
Gadata40[:,0] = np.round( Gadata40[:,0] )

plt.figure()
plt.subplot(311)
plt.title(r'mean CII data of normal subjects')
plt.plot(Gdata25[:,0],Gdata25[:,1],'ko-',label='0.25 mU/kg/min')
plt.plot(Gdata40[:,0],Gdata40[:,1],'kx-',label='0.40 mU/kg/min')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(312)
plt.plot(Idata25[:,0],Idata25[:,1],'ko-',label='0.25 mU/kg/min')
plt.plot(Idata40[:,0],Idata40[:,1],'kx-',label='0.40 mU/kg/min')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.legend()
plt.subplot(313)
plt.plot(Gadata40[:,0],Gadata40[:,1],'kx-',label='0.40 mU/kg/min')
plt.ylabel(r'plasma glucagon [pg/mL]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/cii_healthy.pdf')
plt.close('all')

# --- Euglycemic insulin clamp data

# insulin infusion is scheduled
# glucose infusion is controlled using empirical feedback formula

# mean data of 11 normal subjects
# plasma glucose and plasma insulin data
# page 282

# empirical insulin dose schedule to raise plasma insulin to 100 mU/L
# in 70 kg normal subject is as follows [mU/min] (sampling period is 1 min)
#r_IVI=np.zeros([idx_final,1])
#r_IVI[0]=221
#r_IVI[1]=197
#r_IVI[2]=175
#r_IVI[3]=156
#r_IVI[4]=139
#r_IVI[5]=124
#r_IVI[6]=110
#r_IVI[7]=98
#r_IVI[8]=87
#r_IVI[9]=78
#r_IVI[10:]=69


# glucose infusion [mg/min] adjustment algorithm is as follows:
# (here the sampling period is 5 min)
# r is the infusion rate

#Ts2=5
#k=0
#r=0
#gammac=8.87
#FM=[0, 1]
#SM=[0, 280, 280]

#if t[i]>=Ts2*k:           
#    Gh=x[i,2]
#    if k==0:
#        r=0
#    elif k==1:
#        r=140
#    else:
#        FM[0]=Gb/Gh
#        SM[0]=SM[2]*FM[1]*FM[0]
#        r=gammac*(Gb-Gh)+SM[0]
#        FM=np.roll(FM,1)
#        SM=np.roll(SM,1)  
#    k+=1                 


Gdata=np.loadtxt('G_data_eugc.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt('I_data_eugc.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )

plt.figure()
plt.subplot(211)
plt.title(r'mean euglycemic clamp data of normal subjects')
plt.plot(Gdata[:,0],Gdata[:,1],'ko-')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata[:,0],Idata[:,1],'ko-')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/eugc_healthy.pdf')
plt.close('all')

# --- Deactivation of insulin action data

# experiment following 120 min euglycemic clamp where
# insulin infusion is abruptly stopped and glucose infusion is still
# controlled by same algorithm as in euglycemic clamp

# mean data of 8 normal subjects

# plasma insulin data
# incremental glucose disposal [mg/m2/min] and hepatic glucose output [mg/m2/min data
# page 288

# body surface area of 1.73 m2 was assumed 

# incremental glucose disposal rate = glucose infusion rate  
#                                     + hepatic glucose infusion rate
#                                     - basal hepatic glucose production rate
# 

Idata=np.loadtxt('I_data_ideac.csv', delimiter=',', ndmin=2)
r_IGDdata=np.loadtxt('rIGD_data.csv', delimiter=',', ndmin=2)
r_HGPdata=np.loadtxt('rHGP_data.csv', delimiter=',', ndmin=2)
Idata[:,0] = np.round( Idata[:,0] )
r_IGDdata[:,0] = np.round( r_IGDdata[:,0] )
r_HGPdata[:,0] = np.round( r_HGPdata[:,0] )

plt.figure()
plt.subplot(311)
plt.title(r'mean insulin deactivation data of normal subjects')
plt.plot(Idata[:,0],Idata[:,1],'ko-')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.legend()
plt.subplot(312)
plt.plot(r_IGDdata[:,0],r_IGDdata[:,1],'ko-')
plt.ylabel(r'incremental glucose disposal' +'\n'+ r'[mg/m2/min]',fontsize=fonts)
plt.subplot(313)
plt.plot(r_HGPdata[:,0],r_HGPdata[:,1],'ko-')
plt.ylabel(r'hepatic glucose output' +'\n'+ r'[mg/m2/min]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ideac_healthy.pdf')
plt.close('all')

# --- Hyperglycemic clamp

# glycemia is raised by empirical priming dose of glucose infusion
# after priming dose the glycemia is kept in hyperglycemia steady state
# by variable glucose infusion using empirical feedback formula

# mean data of 11 normal subjects 
# plasma glucose and plasma insulin data
# page 291

# glucose infusion schedule (priming dose) for a +125 mg/dl glycemia raise
# for 70 kg subject is as follows [mg/min] (sampling period is 1 min)

#r_IVG=np.zeros([idx_final,1])
#r_IVG[0]=3059
#r_IVG[1]=2470
#r_IVG[2]=2000
#r_IVG[3]=1588
#r_IVG[4]=1353
#r_IVG[5]=1118
#r_IVG[6]=941
#r_IVG[7]=765
#r_IVG[8]=706
#r_IVG[9]=647


# glucose infusion adjustment algorithm (after priming dose) is as follows:
# (here the sampling period is 5 min)
# r is the infusion rate

#Ts2=5
#k=0
#r=0
#gammac=8.87
#FM=[0, 1]
#SM=[0, 412, 412]
#
#if t[i]>=Ts2*k:           
#    Gh=x[i,2]
#    if k>=2:
#        FM[0]=125/(Gh-Gb)
#        SM[0]=SM[2]*FM[1]*FM[0]
#        r=gammac*(Gb-Gh+125)+SM[0]
#        FM=np.roll(FM,1)
#        SM=np.roll(SM,1)  
#    k+=1   
#
#if i>=10:             
#    r_IVG[i]=r


Gdata=np.loadtxt('G_data_hypergc.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt('I_data_hypergc.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )

plt.figure()
plt.subplot(211)
plt.title(r'mean hyperglycemic clamp data of normal subjects')
plt.plot(Gdata[:,0],Gdata[:,1],'ko-')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata[:,0],Idata[:,1],'ko-')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/hypergc_healthy.pdf')
plt.close('all')


# clinical data used to identify models of metabolic mutlipliers
# extracted from Sorensen's work
# there are also plotted model outputs, using the parameters also from Sorensen's work

# --- hepatic glucose production 
MG_HGPdata=np.loadtxt('MG_HGP_data.csv', delimiter=',', ndmin=2)
GN_L = np.arange(0,4,0.1)
MG_HGP = (1+1.41*np.tanh(0.62 * (1 - 0.497)) ) - 1.41*np.tanh(0.62 * (GN_L - 0.497))  
plt.figure()
plt.subplot(311)
plt.title('Hepatic glucose production - Glucose multiplier')
plt.plot(MG_HGPdata[:,0],MG_HGPdata[:,1],'ko',label='data')
plt.plot(GN_L,MG_HGP,'k',label='model')
plt.ylabel(r'$M_{HGP}^G(t)$ [-]')
plt.xlabel(r'$G_L^N(t)$ [-]')
plt.grid()
MI_HGPdata=np.loadtxt('MI_HGP_data.csv', delimiter=',', ndmin=2)
IN_L = np.arange(0,5,0.1)
MIinf_HGP = (1+1.14*np.tanh(1.66 * (1 - 0.89))) - 1.14*np.tanh(1.66 * (IN_L - 0.89))
plt.subplot(312)
plt.title('Hepatic glucose production - Insulin multiplier')
plt.plot(MI_HGPdata[:,0],MI_HGPdata[:,1],'ko',label='data')
plt.plot(IN_L,MIinf_HGP,'k',label='model')
plt.ylabel(r'$M_{HGP}^{I\infty}(t)$ [-]')
plt.xlabel(r'$I_L^N(t)$ [-]')
plt.grid()
MGa_HGPdata=np.loadtxt('MGa_HGP_data.csv', delimiter=',', ndmin=2)
GaN_L = np.arange(0,5,0.1)
MGa0_HGP = 1/(np.tanh(0.39*1))*np.tanh(0.39*GaN_L)
plt.subplot(313)
plt.title('Hepatic glucose production - Glucagon multiplier')
plt.plot(MGa_HGPdata[:,0],MGa_HGPdata[:,1],'ko',label='data')
plt.plot(GaN_L,MGa0_HGP,'k',label='model')
plt.ylabel(r'$M_{HGP}^{\Gamma}(\infty)$ [-]')
plt.xlabel(r'$\Gamma_L^N(\infty)$ [-]')
plt.grid()
plt.tight_layout()
plt.savefig('Figs/SorensenM_HGP.pdf')
plt.close('all')

# --- hepatic glucose uptake 

MG_HGUdata=np.loadtxt('MG_HGU_data.csv', delimiter=',', ndmin=2)
GN_L = np.arange(0,3,0.1)
a4=5.66
b4=2.44
c4=1.48
MG_HGU= (1-a4*np.tanh(b4 * (1 - c4)) ) + a4*np.tanh(b4 * (GN_L - c4)) 
plt.figure()
plt.title('Hepatic glucose uptake - Glucose multiplier')
plt.plot(MG_HGUdata[:,0],MG_HGUdata[:,1],'ko')
plt.plot(GN_L,MG_HGU,'k')
plt.ylabel(r'$M_{HGU}^G(t)$ [-]')
plt.xlabel(r'$G_L^N(t)$ [-]')
plt.grid()
plt.tight_layout()
plt.savefig('Figs/SorensenM_HGU.pdf')
plt.close('all')


# --- peripheral glucose uptake

MG_PGUdata=np.loadtxt('MG_PGU_data.csv', delimiter=',', ndmin=2)
GN_PI = np.arange(0,4,0.1)
MG_PGU= GN_PI

plt.figure()
plt.subplot(211)
plt.title('Peripheral glucose intake - Glucose multiplier')
plt.plot(MG_PGUdata[:,0],MG_PGUdata[:,1],'ko')
plt.plot(GN_PI,MG_PGU,'k')
plt.ylabel(r'$M_{PGU}^G(t)$ [-]')
plt.xlabel(r'$G_{PI}^N(t)$ [-]')
plt.grid()

MI_PGUdata=np.loadtxt('MI_PGU_data.csv', delimiter=',', ndmin=2)
IN_PI = np.arange(0,11,0.1)

a1=6.52
b1=0.338
c1=5.82

MI_PGU= (1-a1*np.tanh(b1 * (1 - c1))) + a1*np.tanh(b1* (IN_PI - c1))    
plt.subplot(212)
plt.title('Peripheral glucose intake - Insulin multiplier')
plt.plot(MI_PGUdata[:,0],MI_PGUdata[:,1],'ko')
plt.plot(IN_PI,MI_PGU,'k')
plt.ylabel(r'$M_{PGU}^I(t)$ [-]')
plt.xlabel(r'$I_{PI}^N(t)$ [-]')
plt.grid()
plt.tight_layout()
plt.savefig('Figs/SorensenM_PGU.pdf')
plt.close('all')



# --- kidney
GKdata=np.loadtxt('kidney_data.csv', delimiter=',', ndmin=2)
GKs=np.arange(0,50,0.1)
rkge=np.zeros_like(GKs)
index=0

for G_K in GKs:
    if G_K>=0 and G_K<25.5:       
        r_KGE = 0.394 + 0.394*np.tanh(0.198* (G_K - 25.5))    
    elif G_K>=25.5:
        r_KGE = -1.833 + 0.0872 * G_K
    rkge[index]=r_KGE
    index+=1

plt.figure()
plt.title('Kidney glucose extraction')
plt.plot(GKdata[:,0]/18,GKdata[:,1]/180/70,'ko')
plt.plot(GKs,rkge/70,'k')
plt.ylabel(r'$r_{KGE}(t)$ [mmol/min/kg]')
plt.xlabel(r'$G_K(t)$ [mmol/L]')
plt.grid()
plt.tight_layout()
plt.savefig('Figs/Sorensen_KGE.pdf')
plt.close('all')



