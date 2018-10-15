# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:49:14 2018

@author: Admin
"""

# Data published in Dissertation:
# Dynamic Modeling of Glucose Metabolism for the Assessment of T2DM
# by Vahidi 2013
# data were extracted from tables in appendix of the work
# contains ogtt and iivgit data of 10 normal and 10 t2dm subjects together with the bodyweights of individual subjects


# OGTT data:
# 50 g glucose tolerance test
# plasma insulin and glucose data plus incretin data (GLP-1 and GIP)

# IIVGIT
# isoglycemic intravenous glucose infusion test
# glucose infusion was controlled such that plasma glucose profile matched the OGTT data
# plasma insulin and glucose data plus the glucose infusion data

# unit dimensions:

# G [mmol/l]
# I [pmol/L]
# glucose infusion [g]
# GLP-1 and GIP [pmol/L]


#==============================================================================

import numpy as np
from scipy.interpolate import interp1d
import copy
import matplotlib.pyplot as plt
import matplotlib as mpl

# data import
BW_healthy = np.squeeze(np.loadtxt('BW_healthy.csv', ndmin=2))
ivgtt_glu_healthy = np.loadtxt('IVGTT_glucose_healthy.csv', ndmin=2)
ivgtt_glu_healthy = np.insert(ivgtt_glu_healthy,0,np.zeros([1,11]),axis=0)
ivgtt_gly_healthy = np.loadtxt('IVGTT_glycemia_healthy.csv', ndmin=2)
ivgtt_ins_healthy = np.loadtxt('IVGTT_insulin_healthy.csv', ndmin=2)
ogtt_gly_healthy = np.loadtxt('OGTT_glycemia_healthy.csv', ndmin=2)
ogtt_ins_healthy = np.loadtxt('OGTT_insulin_healthy.csv', ndmin=2)
ogtt_glp1_healthy = np.loadtxt('OGTT_GLP1_healthy.csv', ndmin=2)
ogtt_gip_healthy = np.loadtxt('OGTT_GIP_healthy.csv', ndmin=2)

BW_t2dm = np.squeeze(np.loadtxt('BW_t2dm.csv', ndmin=2))
ivgtt_glu_t2dm = np.loadtxt('IVGTT_glucose_t2dm.csv', ndmin=2)
ivgtt_glu_t2dm = np.insert(ivgtt_glu_t2dm,0,np.zeros([1,11]),axis=0)
ivgtt_gly_t2dm = np.loadtxt('IVGTT_glycemia_t2dm.csv', ndmin=2)
ivgtt_ins_t2dm = np.loadtxt('IVGTT_insulin_t2dm.csv', ndmin=2)
ogtt_gly_t2dm = np.loadtxt('OGTT_glycemia_t2dm.csv', ndmin=2)
ogtt_ins_t2dm = np.loadtxt('OGTT_insulin_t2dm.csv', ndmin=2)
ogtt_glp1_t2dm = np.loadtxt('OGTT_GLP1_t2dm.csv', ndmin=2)
ogtt_gip_t2dm = np.loadtxt('OGTT_GIP_t2dm.csv', ndmin=2)


# generation of glucose infusion rates from glucose infusion amounts data
t=np.arange(0,ivgtt_glu_t2dm[-1,0]+5,5)
Rat_t2dm=np.zeros([t.shape[0],10])
Rat_healthy=np.zeros([t.shape[0],10])
#Rat_t2dm0=np.zeros([t.shape[0],10])
#Rat_healthy0=np.zeros([t.shape[0],10])
for j in range(0,10):
    f=interp1d(ivgtt_glu_t2dm[:,0],np.cumsum(ivgtt_glu_t2dm[:,j+1]), kind='linear')
    temp=f(t)
    temp[0]=0
    Rat_t2dm[:,j]=np.gradient(temp,5)
#    Rat_t2dm[0,j]=0
    
#    for i in range(len(t)):
#        Rat_t2dm0[i,j]=(temp[i]-temp[i-1])/5
#    Rat_t2dm0[0,j]=0


#    ii=1
#    for i in range(t.shape[0]):
#        ii=np.min([ii,ivgtt_glu_t2dm[:,0].shape[0]-1])
#        if t[i]<ivgtt_glu_t2dm[ii,0]:
#            Rat_t2dm0[i,j]=ivgtt_glu_t2dm[ii,j+1]/(ivgtt_glu_t2dm[ii,0]-ivgtt_glu_t2dm[ii-1,0])
#        else: 
#            ii+=1
#            ii=np.min([ii,ivgtt_glu_t2dm[:,0].shape[0]-1])
#            Rat_t2dm0[i,j]=ivgtt_glu_t2dm[ii,j+1]/(ivgtt_glu_t2dm[ii,0]-ivgtt_glu_t2dm[ii-1,0])
#

                 
    f=interp1d(ivgtt_glu_healthy[:,0],np.cumsum(ivgtt_glu_healthy[:,j+1]), kind='linear')
    temp=f(t)
    temp[0]=0
    Rat_healthy[:,j]=np.gradient(temp,5)
#    Rat_healthy[0,j]=0

#    for i in range(len(t)):
#        Rat_healthy0[i,j]=(temp[i]-temp[i-1])/5
#    Rat_healthy0[0,j]=0               


#    ii=1
#    for i in range(t.shape[0]):
#        ii=np.min([ii,ivgtt_glu_healthy[:,0].shape[0]-1])
#        if t[i]<ivgtt_glu_healthy[ii,0]:
#            Rat_healthy0[i,j]=ivgtt_glu_healthy[ii,j+1]/(ivgtt_glu_healthy[ii,0]-ivgtt_glu_healthy[ii-1,0])
#        else: 
#            ii+=1
#            ii=np.min([ii,ivgtt_glu_healthy[:,0].shape[0]-1])
#            Rat_healthy0[i,j]=ivgtt_glu_healthy[ii,j+1]/(ivgtt_glu_healthy[ii,0]-ivgtt_glu_healthy[ii-1,0])        


# generation of incretin data (sum of GIP-1 and GLP)
# healthy  
ogtt_incretin_healthy = ogtt_glp1_healthy + ogtt_gip_healthy
ogtt_incretin_healthy[:,0]=ogtt_glp1_healthy[:,0]
ogtt_incretin0_healthy=copy.deepcopy(ogtt_incretin_healthy)
ogtt_incretin0_healthy[:,1:]=ogtt_incretin0_healthy[:,1:]-ogtt_incretin0_healthy[0,1:]
# t2dm
ogtt_incretin_t2dm = ogtt_glp1_t2dm + ogtt_gip_t2dm
ogtt_incretin_t2dm[:,0]=ogtt_glp1_t2dm[:,0]
ogtt_incretin0_t2dm=copy.deepcopy(ogtt_incretin_t2dm)
ogtt_incretin0_t2dm[:,1:]=ogtt_incretin0_t2dm[:,1:]-ogtt_incretin0_t2dm[0,1:]


# ******** mean and standard deviation values

# healthy people
BW_healthy_mean=np.mean(BW_healthy)
BW_healthy_std=np.std(BW_healthy)
Rat_healthy_mean=np.mean(Rat_healthy,1)
Rat_healthy_std=np.std(Rat_healthy,1)
#Rat_healthy0_mean=np.mean(Rat_healthy0,1)
ivgtt_gly_healthy_mean=np.mean(ivgtt_gly_healthy[:,1:],1)
ivgtt_gly_healthy_std=np.std(ivgtt_gly_healthy[:,1:],1)
ivgtt_ins_healthy_mean=np.mean(ivgtt_ins_healthy[:,1:],1)
ivgtt_ins_healthy_std=np.std(ivgtt_ins_healthy[:,1:],1)

ogtt_gly_healthy_mean=np.mean(ogtt_gly_healthy[:,1:],1)
ogtt_gly_healthy_std=np.std(ogtt_gly_healthy[:,1:],1)
ogtt_ins_healthy_mean=np.mean(ogtt_ins_healthy[:,1:],1)
ogtt_ins_healthy_std=np.std(ogtt_ins_healthy[:,1:],1)
ogtt_glp1_healthy_mean=np.mean(ogtt_glp1_healthy[:,1:],1)
ogtt_gip_healthy_mean=np.mean(ogtt_gip_healthy[:,1:],1)

ogtt_incretin_healthy_mean=np.mean(ogtt_incretin_healthy[:,1:],1)
ogtt_incretin_healthy_std=np.std(ogtt_incretin_healthy[:,1:],1)

ogtt_incretin0_healthy_mean=np.mean(ogtt_incretin0_healthy[:,1:],1)
ogtt_incretin0_healthy_std=np.std(ogtt_incretin0_healthy[:,1:],1)

# t2dm diabetics 
BW_t2dm_mean=np.mean(BW_t2dm)
BW_t2dm_std=np.std(BW_t2dm)
Rat_t2dm_mean=np.mean(Rat_t2dm,1)
Rat_t2dm_std=np.std(Rat_t2dm,1)
#Rat_t2dm0_mean=np.mean(Rat_t2dm0,1)
ivgtt_gly_t2dm_mean=np.mean(ivgtt_gly_t2dm[:,1:],1)
ivgtt_gly_t2dm_std=np.std(ivgtt_gly_t2dm[:,1:],1)
ivgtt_ins_t2dm_mean=np.mean(ivgtt_ins_t2dm[:,1:],1)
ivgtt_ins_t2dm_std=np.std(ivgtt_ins_t2dm[:,1:],1)

ogtt_gly_t2dm_mean=np.mean(ogtt_gly_t2dm[:,1:],1)
ogtt_gly_t2dm_std=np.std(ogtt_gly_t2dm[:,1:],1)
ogtt_ins_t2dm_mean=np.mean(ogtt_ins_t2dm[:,1:],1)
ogtt_ins_t2dm_std=np.std(ogtt_ins_t2dm[:,1:],1)
ogtt_glp1_t2dm_mean=np.mean(ogtt_glp1_t2dm[:,1:],1)
ogtt_gip_t2dm_mean=np.mean(ogtt_gip_t2dm[:,1:],1)

ogtt_incretin_t2dm_mean=np.mean(ogtt_incretin_t2dm[:,1:],1)
ogtt_incretin_t2dm_std=np.std(ogtt_incretin_t2dm[:,1:],1)

ogtt_incretin0_t2dm_mean=np.mean(ogtt_incretin0_t2dm[:,1:],1)
ogtt_incretin0_t2dm_std=np.std(ogtt_incretin0_t2dm[:,1:],1)

# data plot and save

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

sirka=9
vyska=4*3
fonts=8
marks=3

plt.figure(figsize=(cm2inch(sirka), cm2inch(vyska)))
plt.subplot(311)
plt.title(r'mean OGTT data of normal subjects')
plt.errorbar(ogtt_gly_healthy[:,0],ogtt_gly_healthy_mean,yerr=ogtt_gly_healthy_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma glucose [mmol/L]',fontsize=fonts)
plt.subplot(312)
plt.errorbar(ogtt_ins_healthy[:,0],ogtt_ins_healthy_mean,yerr=ogtt_ins_healthy_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma insulin[pmol/L]',fontsize=fonts)
plt.subplot(313)
plt.errorbar(ogtt_incretin_healthy[:,0],ogtt_incretin_healthy_mean,yerr=ogtt_incretin_healthy_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'incretins [pmol/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ogtt_healthy.pdf')
plt.close('all')


plt.figure(figsize=(cm2inch(sirka), cm2inch(vyska)))
plt.subplot(311)
plt.title(r'mean OGTT data of t2dm subjects')
plt.errorbar(ogtt_gly_t2dm[:,0],ogtt_gly_t2dm_mean,yerr=ogtt_gly_t2dm_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma glucose [mmol/L]',fontsize=fonts)
plt.subplot(312)
plt.errorbar(ogtt_ins_t2dm[:,0],ogtt_ins_t2dm_mean,yerr=ogtt_ins_t2dm_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma insulin[pmol/L]',fontsize=fonts)
plt.subplot(313)
plt.errorbar(ogtt_incretin_t2dm[:,0],ogtt_incretin_t2dm_mean,yerr=ogtt_incretin_t2dm_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'incretins [pmol/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ogtt_t2dm.pdf')
plt.close('all')


plt.figure(figsize=(cm2inch(sirka), cm2inch(vyska)))
plt.subplot(311)
plt.title(r'mean IIVGIT data of normal subjects')
plt.errorbar(ivgtt_gly_healthy[:,0],ivgtt_gly_healthy_mean,yerr=ivgtt_gly_healthy_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma glucose [mmol/L]',fontsize=fonts)
plt.subplot(312)
plt.errorbar(ivgtt_ins_healthy[:,0],ivgtt_ins_healthy_mean,yerr=ivgtt_ins_healthy_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma insulin[pmol/L]',fontsize=fonts)
plt.subplot(313)
plt.plot(t,Rat_healthy_mean,'k')
plt.fill_between(t,Rat_healthy_mean-Rat_healthy_std,Rat_healthy_mean+Rat_healthy_std,facecolor='grey',linewidth=0.0)
plt.ylabel(r'glucose infusion rate [g/min]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/iivgit_healthy.pdf')
plt.close('all')

plt.figure(figsize=(cm2inch(sirka), cm2inch(vyska)))
plt.subplot(311)
plt.title(r'mean IIVGIT data of t2dm subjects')
plt.errorbar(ivgtt_gly_t2dm[:,0],ivgtt_gly_t2dm_mean,yerr=ivgtt_gly_t2dm_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma glucose [mmol/L]',fontsize=fonts)
plt.subplot(312)
plt.errorbar(ivgtt_ins_t2dm[:,0],ivgtt_ins_t2dm_mean,yerr=ivgtt_ins_t2dm_std,fmt='ko-',capsize=3,markersize=marks)
plt.ylabel(r'plasma insulin[pmol/L]',fontsize=fonts)
plt.subplot(313)
plt.plot(t,Rat_t2dm_mean,'k')
plt.fill_between(t,Rat_t2dm_mean-Rat_t2dm_std,Rat_t2dm_mean+Rat_t2dm_std,facecolor='grey',linewidth=0.0)
plt.ylabel(r'glucose infusion rate [g/min]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/iivgit_t2dm.pdf')
plt.close('all')