# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:45:11 2018

@author: Admin
"""

# Data published in paper:
# Insulin Sensitivity from Meal Tolerance Tests in Normal Subjects A Minimal Model Index
# by Caumo et. al. 2000
# all data were extracted from graphs
# contains glucose and insulin data of IVGTT and MTT
# together 10 normal subjects

# IVGTT - (possibly) 1 min 0.3 g/kg with insulin injection 30 mU/kg at 20 min (over 1 min?)
# MTT - reported 75 g carbohydrate content eaten within 10 min

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


# --- IVGTT
Gdata=np.loadtxt('G_data_ivgtt_Bergman.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt('I_data_ivgtt_Bergman.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )

plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT (1 min 0.3g/kg) data')
plt.plot(Gdata[:,0],Gdata[:,1],'ko-')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata[:,0],Idata[:,1],'ko-')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ivgtt_data.pdf')
plt.close('all')


# --- MTT
Gdata=np.loadtxt('G_data_mgtt_Bergman.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt('I_data_mgtt_Bergman.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )

plt.figure()
plt.subplot(211)
plt.title(r'mean MTT (75 g) data')
plt.plot(Gdata[:,0],Gdata[:,1],'ko-')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata[:,0],Idata[:,1],'ko-')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/mtt_data.pdf')
plt.close('all')
