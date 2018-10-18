# demo of Sorensen model simulation of IVGTT (intravenous glugose tolerance) test
# simulation of normal subject

import numpy as np
import matplotlib.pyplot as plt
from fcns_Sorensen import sim_Sorensen


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

# IVGTT data import
path='../Data/Sorensen 1978 - IVGTT IVITT Clamps/'
Gdata=np.loadtxt(path+'G_data.csv', delimiter=',', ndmin=2)
Idata=np.loadtxt(path+'I_data.csv', delimiter=',', ndmin=2)
Gdata[:,0] = np.round( Gdata[:,0] )
Idata[:,0] = np.round( Idata[:,0] )

# glucose dose
g_dose = 0.5 # [g/kg]
glu_bolus = g_dose*1e3/180 # [mmol/kg]
glu_bolus_min = 3 

# simulation time
Ts=1
tsim=Gdata[-1,0]
t=np.arange(Gdata[0,0],tsim+Ts,Ts)
idx_final=t.size

# generating the inputs for glucose and insulin 
r_IVG=np.zeros([idx_final,1])
r_IVI=np.zeros([idx_final,1])
r_IVG[0:int(glu_bolus_min/Ts),0]=glu_bolus/Ts/glu_bolus_min

# basal glucose and insulin states
Gb=Gdata[0,1]/0.8/18
Ib=Idata[0,1]

# import of identified parameters
p=np.loadtxt('Sorensen_h_parnew.csv')

# simulation
x=sim_Sorensen(t,p,Gb,Ib,r_IVG,r_IVI)
Gsim=x[:,6]*0.8
Isim=x[:,16]

# comparing simulation output with data
Gsim2=resamp(t,Gdata[:,0],Gsim)
Isim2=resamp(t,Idata[:,0],Isim)
rsqG=r_squared(Gdata[:,1]/18,Gsim2)
rsqI=r_squared(Idata[:,1],Isim2)

# final plot
plt.figure()
plt.subplot(211)
plt.title(r'blood glucose in peripheries, $R^2$='+str(np.round(rsqG,2)))
plt.plot(t,Gsim,'k')
plt.plot(Gdata[:,0],Gdata[:,1]/18,'ko')
plt.ylabel(r'$G(t)$ [mmol/L]')
plt.xlabel(r'time [min]')
plt.grid()
plt.subplot(212)
plt.title(r'plasma insulin in peripheries, $R^2$='+str(np.round(rsqI,2)))
plt.plot(t,Isim,'k')
plt.plot(Idata[:,0],Idata[:,1],'ko')
plt.ylabel(r'$I(t)$ [mU/L]')
plt.xlabel(r'time [min]')
plt.grid()
plt.tight_layout()
plt.savefig('Sorensen_demo_output.pdf')
plt.close('all')
