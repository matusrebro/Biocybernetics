# -*- coding: utf-8 -*-
"""
Created on Thu May 10 13:48:13 2018

@author: Admin
"""
# Data published in paper:
# Physiologic Evaluation of Factors Controlling Glucose Tolerance in Man
# by Bergman et. al. 1981
# all data were extracted from graphs
# contains glucose and insulin data from 1-min 300 mg/kg IVGTT experiments 
# together 18 subjects (8 lean and 10 obese)
# datasets were further divided into 4 groups:
# 1. normal tolerance lean (n=5)
# 2. low tolerance lean (n=3)
# 3. normal tolerance obese (n=3)
# 4. low tolerance obese (n=7)

# unit dimensions:

# G [mg/dl]
# I [mU/L]


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

sirka=9
vyska=4*3
fonts=8
marks=3


# --- lean subjects
Gdata1=np.loadtxt('G_data_ivgtt_Bergman_group1.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt('I_data_ivgtt_Bergman_group1.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Gdata1[0,0] = 0
Idata1[0,0] = 0

Gdata2=np.loadtxt('G_data_ivgtt_Bergman_group2.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt('I_data_ivgtt_Bergman_group2.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )
Gdata2[0,0] = 0
Idata2[0,0] = 0


plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT (1 min 0.3g/kg) data of lean subjects')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko-',label='normal tolerance')
plt.plot(Gdata2[:,0],Gdata2[:,1],'kx-',label='low tolerance')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata1[:,0],Idata1[:,1],'ko-',label='normal tolerance')
plt.plot(Idata2[:,0],Idata2[:,1],'kx-',label='low tolerance')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ivgtt_lean.pdf')
plt.close('all')

# --- obese subjects

Gdata1=np.loadtxt('G_data_ivgtt_Bergman_group3.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt('I_data_ivgtt_Bergman_group3.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Gdata1[0,0] = 0
Idata1[0,0] = 0

Gdata2=np.loadtxt('G_data_ivgtt_Bergman_group4.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt('I_data_ivgtt_Bergman_group4.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )
Gdata2[0,0] = 0
Idata2[0,0] = 0


plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT (1 min 0.3g/kg) data of obese subjects')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko-',label='normal tolerance')
plt.plot(Gdata2[:,0],Gdata2[:,1],'kx-',label='low tolerance')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata1[:,0],Idata1[:,1],'ko-',label='normal tolerance')
plt.plot(Idata2[:,0],Idata2[:,1],'kx-',label='low tolerance')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ivgtt_obese.pdf')
plt.close('all')