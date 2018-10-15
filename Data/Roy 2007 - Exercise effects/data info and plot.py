# -*- coding: utf-8 -*-
"""
Created on Thu May 10 15:20:45 2018

@author: Admin
"""

# Data published in paper:
# Dynamic Modeling of Exercise Effects on Plasma Glucose and Insuliin levels
# by Roy et. al. 2007
# all data were extracted from graphs
# contains glucose and insulin data extracted during physical exercise

# healthy subjects (how many?) performed mild exercise (PVOmax=40) lasting 60 minutes

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


Gupdata=np.loadtxt('Gupdata1.csv', delimiter=',', ndmin=2)
Gproddata=np.loadtxt('Gproddata1.csv', delimiter=',', ndmin=2)
Gupdata[:,0] = np.round( Gupdata[:,0] )
Gproddata[:,0] = np.round( Gproddata[:,0] )

Gdata=np.loadtxt('Gexdata.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt('Iexdata.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )

# Glucose uptake and hepatic glucose production
# Deviation form

plt.figure()
plt.subplot(211)
plt.title(r'mean glucose uptake and production data')
plt.plot(Gupdata[:,0],Gupdata[:,1],'ko-')
plt.ylabel(r'Gup [mg/kg/min]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Gproddata[:,0],Gproddata[:,1],'ko-')
plt.ylabel(r'Gprod [mg/kg/min]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/Ex_data1.pdf')
plt.close('all')

plt.figure()
plt.subplot(211)
plt.title(r'mean glucose and insulin data')
plt.plot(Gdata[:,0],Gdata[:,1],'ko-')
plt.ylabel(r'plasma glucose [mg/dL]',fontsize=fonts)
plt.legend()
plt.subplot(212)
plt.plot(Idata[:,0],Idata[:,1],'ko-')
plt.ylabel(r'plasma insulin [mU/L]',fontsize=fonts)
plt.xlabel('t [minutes]')
plt.tight_layout()
plt.savefig('Figs/Ex_data2.pdf')
plt.close('all')