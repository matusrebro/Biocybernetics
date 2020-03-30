import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from adaptive_control_fcns import sim_MRAC

# import of carbohydrate intake data
DataSetName = 'Dat_test_4days'
Dat_carb = np.loadtxt('Data/'+DataSetName + '_carb.csv', delimiter=',', ndmin=2) # 10 g

Ts = 5
t_start = 0
t_stop = 4*24*60
idx_final = int((t_stop - t_start)/Ts) + 1

# main time vector:
tt = np.zeros([idx_final, 1])

for idx in range(1,idx_final):
    tt[idx]=idx*Ts
    

# input signal generator: (initial inputs and also whole input signal)
dsig = np.zeros([idx_final, 1]) # mmol/min
for carbRow in Dat_carb:
    dsig[int(carbRow[0]/Ts), 0] = (carbRow[1] *10)/Ts

# Hovorka model parameters
Hp=np.load('paro02c.npy')

t_I, V_I, k_I, A_G, t_G, k_12, V_G, EGP_0, F_01, k_b1, k_b2, k_b3, k_a1, k_a2, k_a3, t_cgm = Hp

Gb=7

# reference signal
ra=0.5
rper=24*60
r=-ra*signal.square(tt/rper*2*np.pi)

dsigm=dsig*1000/180 # carbohydrate intake as seen by model (insulin-glucose system)
dsigc=dsig*Ts # this signal is for disturbance rejection algorithm

# adaptive control simulation
x, u, ud, vb = sim_MRAC(tt,Hp,dsigm,dsigc,r,Gb)

# signals for plotting
Cdata=dsig*Ts # carbohydrates [g]
Vbasdata=(u+vb)/1e3*60 # basal insulin [U/h]
Vboldata=-ud/1e3*Ts # bolus insulin [U]
Vboldata=np.abs(Vboldata)

Gcgm=x[:,10]
plt.figure()
plt.subplot(311)
plt.xlim([0,4.1])
plt.plot(tt/60/24,Gcgm,'k')
plt.plot([0,tt[-1]/60/24],[4,4],'k--')
plt.plot([0,tt[-1]/60/24],[10,10],'k--')
plt.title(r'Glycemia')
plt.xlabel(r'time [days]')
plt.ylabel(r'$G$ [mmol/L]')

plt.subplot(312)
plt.title(r'Carbohydrates')
plt.xlim([0,4.1])
plt.stem(tt[Cdata>0]/60/24,Cdata[Cdata>0],'k',markerfmt='ko', basefmt='k')
plt.xlabel(r'time [days]')
plt.ylabel('C [g]')

plt.subplot(313)
plt.title(r'Basal (full line) and bolus (stem) insulin')
plt.xlim([0,4.1])
plt.plot(tt/60/24,Vbasdata,'k')
plt.xlabel(r'time [days]')
plt.ylabel(r'Basal [U/h]')
plt.twinx()
plt.xlim([0,4.1])
plt.stem(tt[Vboldata>0]/60/24,Vboldata[Vboldata>0],'k',markerfmt='ko', basefmt='k')
plt.xlabel(r'time [days]')
plt.ylabel(r'Bolus [U]')
plt.savefig('MRACdemo.pdf')
plt.close('all')


