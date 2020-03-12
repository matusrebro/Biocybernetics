# -*- coding: utf-8 -*-
"""
Created on Thu May 10 14:18:10 2018

@author: Admin
"""

# Data published in paper:
# Quantitative Insulin Sensitivity Check Index
# by Katz et. al. 2000
# all data were extracted from graphs
# contains glucose and insulin data of IVGTT and hyperinsulemic clamp
# of 28 nonobese, 13 obese and 15 t2dm subjects

# IVGTT - 2 min 0.3 g/kg glucose and 20 min after that 4 mU/kg/min for 5 min (20 mU/kg over 5 min)
# Clamp - there is only mentioned 120 mU/m2/min insulin infusion
#         glucose infusion probably controlled by classic algorithm

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


# IVGTT

Gdata1=np.loadtxt('G_data_fsivgtt_h.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt('I_data_fsivgtt_h.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Gdata1[0,0] = 0
Idata1[0,0] = 0

Gdata2=np.loadtxt('G_data_fsivgtt_o.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt('I_data_fsivgtt_o.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )
Gdata2[0,0] = 0
Idata2[0,0] = 0

Gdata3=np.loadtxt('G_data_fsivgtt_d.csv', delimiter=',', ndmin=2)
Idata3=np.loadtxt('I_data_fsivgtt_d.csv', delimiter=',', ndmin=2)
Gdata3[:,0] = np.round( Gdata3[:,0] )
Idata3[:,0] = np.round( Idata3[:,0] )
Gdata3[0,0] = 0
Idata3[0,0] = 0

plt.figure()
plt.subplot(211)
plt.title(r'mean IVGTT data of normal, obese and t2dm subjects')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko-',label='normal')
plt.plot(Gdata2[:,0],Gdata2[:,1],'kx-',label='obese')
plt.plot(Gdata3[:,0],Gdata3[:,1],'ks-',label='t2dm')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata1[:,0],Idata1[:,1],'ko-',label='normal')
plt.plot(Idata2[:,0],Idata2[:,1],'kx-',label='obese')
plt.plot(Idata3[:,0],Idata3[:,1],'ks-',label='t2dm')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/ivgtt_data.pdf')
plt.close('all')

# isoglycemic hyperinsulemic clamp

Gdata1=np.loadtxt('G_data_Gclamp_h.csv', delimiter=',', ndmin=2)
Idata1=np.loadtxt('I_data_Gclamp_h.csv', delimiter=',', ndmin=2)
Radata1=np.loadtxt('Ra_data_Gclamp_h.csv', delimiter=',', ndmin=2)
Gdata1[:,0] = np.round( Gdata1[:,0] )
Idata1[:,0] = np.round( Idata1[:,0] )
Radata1[:,0] = np.round( Radata1[:,0] )
Gdata1[0,0] = 0
Idata1[0,0] = 0

Gdata2=np.loadtxt('G_data_Gclamp_o.csv', delimiter=',', ndmin=2)
Idata2=np.loadtxt('I_data_Gclamp_o.csv', delimiter=',', ndmin=2)
Radata2=np.loadtxt('Ra_data_Gclamp_o.csv', delimiter=',', ndmin=2)
Gdata2[:,0] = np.round( Gdata2[:,0] )
Idata2[:,0] = np.round( Idata2[:,0] )
Radata2[:,0] = np.round( Radata2[:,0] )
Gdata2[0,0] = 0
Idata2[0,0] = 0

Gdata3=np.loadtxt('G_data_Gclamp_d.csv', delimiter=',', ndmin=2)
Idata3=np.loadtxt('I_data_Gclamp_d.csv', delimiter=',', ndmin=2)
Radata3=np.loadtxt('Ra_data_Gclamp_d.csv', delimiter=',', ndmin=2)
Gdata3[:,0] = np.round( Gdata3[:,0] )
Idata3[:,0] = np.round( Idata3[:,0] )
Radata3[:,0] = np.round( Radata3[:,0] )
Gdata3[0,0] = 0
Idata3[0,0] = 0

plt.figure()
plt.subplot(311)
plt.title(r'mean glucose clamp data of normal, obese and t2dm subjects')
plt.plot(Gdata1[:,0],Gdata1[:,1],'ko-',label='normal')
plt.plot(Gdata2[:,0],Gdata2[:,1],'kx-',label='obese')
plt.plot(Gdata3[:,0],Gdata3[:,1],'ks-',label='t2dm')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(312)
plt.plot(Idata1[:,0],Idata1[:,1],'ko-',label='normal')
plt.plot(Idata2[:,0],Idata2[:,1],'kx-',label='obese')
plt.plot(Idata3[:,0],Idata3[:,1],'ks-',label='t2dm')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.subplot(313)
plt.plot(Radata1[:,0],Radata1[:,1],'ko-',label='normal')
plt.plot(Radata2[:,0],Radata2[:,1],'kx-',label='obese')
plt.plot(Radata3[:,0],Radata3[:,1],'ks-',label='t2dm')
plt.ylabel(r'glucose infusion [mg/min]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/clamp_data.pdf')
plt.close('all')
