# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:45:11 2018

@author: Admin
"""

# Data published in paper:
# Effect of Moderate Exercise Training on Peripheral Glucose Effectiveness, Insulin Sensitivity
# by Nishida et. al. 2004
# all data were extracted from graphs
# contains glucose and insulin data of IVGTT before, 16h and 1 week after
# 12 week training program
# exercise consisted of 60 min/day cycling 5 days/week at lactate threshold intensity
# together 8 normal subjects

# IVGTT - 1 min 0.3 g/kg with insulin injection 20 mU/kg at 20 min (over 5 min)


# unit dimensions:

# G [mg/dL]
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


# --- IVGTT
Gdata1=np.loadtxt('G_data_ivgtt_Nishida1.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt('I_data_ivgtt_Nishida1.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Gdata2=np.loadtxt('G_data_ivgtt_Nishida2.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt('I_data_ivgtt_Nishida2.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )
Gdata3=np.loadtxt('G_data_ivgtt_Nishida3.csv', delimiter=',', ndmin=2)
Idata3=np.loadtxt('I_data_ivgtt_Nishida3.csv', delimiter=',', ndmin=2)
Gdata3[:,0] = np.round( Gdata3[:,0] )
Idata3[:,0] = np.round( Idata3[:,0] )



plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT (1 min 0.3g/kg with 5 min 20mU/kg at 20 min) data')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko-',label='before training')
plt.plot(Gdata2[:,0],Gdata2[:,1],'kx-',label='16h after training')
plt.plot(Gdata3[:,0],Gdata3[:,1],'ks-',label='1w after training')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata1[:,0],Idata1[:,1],'ko-',label='before training')
plt.plot(Idata2[:,0],Idata2[:,1],'kx-',label='16h after training')
plt.plot(Idata3[:,0],Idata3[:,1],'ks-',label='1w after training')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.legend()
plt.tight_layout()
plt.savefig('Figs/ivgtt_data.pdf')
plt.close('all')

