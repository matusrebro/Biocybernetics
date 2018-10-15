# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:45:11 2018

@author: Admin
"""

# Data published in paper:
# Influence of Short-Term Submaximal Exercise on Parameters of Glucose Assimilation Analyzed With the Minimal Model
# by Brun et. al. 1995
# all data were extracted from graphs
# contains glucose and insulin data of IVGTT at rest and after exercise
# exercise consisted of cycling with an increasing load for 5 minutes 
# followed by 10 minutes at theoretical 85 percent of maximal heart rate
# ivgtt was performed 25 after exercise
# together 7 normal subjects

# IVGTT - 3 min 0.5 g/kg with insulin injection 20 mU/kg at 20 min (over 1 min?)


# unit dimensions:

# G [mmol/L]
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
Gdata1=np.loadtxt('G_data_ivgtt_Brun1.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt('I_data_ivgtt_Brun1.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Gdata2=np.loadtxt('G_data_ivgtt_Brun2.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt('I_data_ivgtt_Brun2.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )

plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT (3 min 0.5g/kg with 1 min 20mU/kg at 20 min) data')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko-',label='at rest')
plt.plot(Gdata2[:,0],Gdata2[:,1],'kx-',label='after exercise')
plt.ylabel(r'plasma glucose [mmol/L]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata1[:,0],Idata1[:,1],'ko-',label='at rest')
plt.plot(Idata2[:,0],Idata2[:,1],'kx-',label='after exercise')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.legend()
plt.tight_layout()
plt.savefig('Figs/ivgtt_data.pdf')
plt.close('all')

