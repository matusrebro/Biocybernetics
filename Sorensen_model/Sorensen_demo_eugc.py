# demo of Sorensen model simulation of IVGTT (intravenous glugose tolerance) test
# simulation of normal subject

import numpy as np
import matplotlib.pyplot as plt
from fcns_Sorensen import sim_Sorensen_eugc


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


# euglycemic clamp data import
path='../Data/Sorensen 1978 - IVGTT IVITT Clamps/'
Gdata=np.loadtxt(path+'G_data_eugc.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt(path+'I_data_eugc.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )

# simulation time
Ts=1
tsim=Gdata[-1,0]
t=np.arange(Gdata[0,0],tsim+Ts,Ts)
idx_final=t.size

# loading of model parameters
p=np.loadtxt('Sorensen_h_parnew.csv')

# we simulate average 70 kg subject
BW=70

# generating the insulin infusion input signal to achieve the hyperinsulinemia
r_IVI=np.zeros(idx_final)
r_IVI[0:10]=np.array([221, 197, 175, 156, 139, 124, 110, 98, 87, 78])/BW
r_IVI[10:]=69/70

# basal glucose and insulin states
Gb=Gdata[0,1]/0.925/18
Ib=Idata[0,1]

# import of identified parameters
p=np.loadtxt('Sorensen_h_parnew.csv')

# simulation
x,r_IVGsim=sim_Sorensen_eugc(t,p,Gb,Ib,r_IVI,BW)
Gsim=x[:,2]*0.925
Isim=x[:,12]

# comparing simulation output with data
Gsim2=resamp(t,Gdata[:,0],Gsim)
Isim2=resamp(t,Idata[:,0],Isim)
rsqG=r_squared(Gdata[:,1]/18,Gsim2)
rsqI=r_squared(Idata[:,1],Isim2)


plt.figure()
plt.subplot(311)
plt.title(r'plasma glucose in arteries, $R^2$='+str(np.round(rsqG,2)))
plt.plot(t,x[:,2]*0.925,'k')
plt.plot(Gdata[:,0],Gdata[:,1]/18,'ko')
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'čas [min]')
plt.grid()
plt.subplot(312)
plt.title(r'plasma insulin in arteries, $R^2$='+str(np.round(rsqI,2)))
plt.plot(t,x[:,12],'k')
plt.plot(Idata[:,0],Idata[:,1],'ko')
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'čas [min]')
plt.grid()
plt.subplot(313)
plt.title(r'glucose and insulin infusion')
plt.plot(t,r_IVGsim,'k',label=r'$r_{IVG}(t)$')
plt.ylabel(r'$r_{IVG}(t)$ [mmol/kg/min]')
plt.legend(loc=(0.75,0.6))
plt.twinx()
plt.plot(t,r_IVI,'k--',label=r'$r_{IVI}(t)$')
plt.ylabel(r'$r_{IVI}(t)$ [mU/kg/min]')
plt.xlabel(r'čas [min]')
plt.legend(loc=(0.75,0.45))
plt.grid()
plt.tight_layout()
plt.savefig('Sorensen_eugc_demo_output.pdf')
plt.close('all')

