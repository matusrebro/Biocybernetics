

import numpy as np
import time, datetime
import pandas as pd
import matplotlib.pyplot as plt

G_data1 = pd.read_csv('Data/Timestamp_GlucoseSensor_RawData.csv',names=['Datetime', 'CGM'], header=None, parse_dates=True)
D_data1 = pd.read_csv('Data/Timestamp_Carb_RawData.csv',names=['Datetime', 'Carbs'], header=None, parse_dates=True)
Vbas_data1 = pd.read_csv('Data/Timestamp_Basal_RawData.csv',names=['Datetime', 'Basal'], header=None, parse_dates=True)
Vbol_data1 = pd.read_csv('Data/Timestamp_Bolus_RawData.csv',names=['Datetime', 'Bolus'], header=None, parse_dates=True)

Ddt=D_data1.Datetime
Ddt=pd.to_datetime(Ddt)
Gdt=G_data1.Datetime
Gdt=pd.to_datetime(Gdt)
Basdt=Vbas_data1.Datetime
Basdt=pd.to_datetime(Basdt)
Boldt=Vbol_data1.Datetime
Boldt=pd.to_datetime(Boldt)


refc=[Ddt[0], Gdt[0], Basdt[0], Boldt[0]]    

reft=refc[np.argmin(refc)]  

def ts2time(data,reft):
    reft=time.mktime(reft.timetuple())/60
    vec=[]    
    for line in data:
        vec.append(time.mktime(line.timetuple())/60-reft)
    return vec


ttD=ts2time(Ddt,reft)
ttG=ts2time(Gdt,reft)
ttBas=ts2time(Basdt,reft)
ttBol=ts2time(Boldt,reft)


# time vector
Ts=5
t_stop = np.max([ttD[-1], ttG[-1], ttBas[-1], ttBol[-1]])
idx_final = int(t_stop/Ts) + 1
tt = np.zeros([idx_final, 1])
for idx in range(1,idx_final):
    tt[idx]=idx*Ts

ttd=[]
ttd.append(reft)
for cas in tt[1:]:
    ttd.append(reft+datetime.timedelta(minutes=int(cas)))
    
    
# CGM data interpolation
Graw=G_data1.CGM
G=np.interp(tt,ttG,Graw)

#plt.plot(tt/60/24,G)
#plt.plot(np.asarray(ttG)/60/24,Graw,'ro')

# generation of d(t) [g/min] signal - rate of carbohydrate intake
d=D_data1.Carbs
CarbTable=np.vstack((ttD,d))
CarbTable=np.transpose(CarbTable)
dsig = np.zeros([idx_final, 1])
for carbRow in CarbTable:
    dsig[int(carbRow[0]/Ts), 0] = (carbRow[1] *10)/Ts
    
#plt.plot(tt,dsig)

bas=Vbas_data1.Basal
bol=Vbol_data1.Bolus
BasTable=np.vstack((ttBas,bas))
BasTable=np.transpose(BasTable)
BolTable=np.vstack((ttBol,bol))
BolTable=np.transpose(BolTable)

#plt.plot(np.asarray(ttBas)/60 ,bas)

vbas = np.zeros([idx_final, 1])
for idx in range(vbas.shape[0]):
    lastValue = BasTable[:,1][ BasTable[:,0]<=idx*Ts ][-1]
    vbas[idx] = lastValue * ( 10**3/60)

#plt.plot(tt,vbas)

vbol = np.zeros([idx_final, 1])
for bolusRow in BolTable:
    vbol[int(bolusRow[0]/Ts), 0] = (bolusRow[1] * 1e3)/Ts

#plt.plot(tt,vbol)

vb=np.mean(vbas)

vsig = vbol + vbas #v_b
##plt.plot(tt,vsig)

idx_final=tt.shape[0]


from Gpredfun import Gpredarx2, clarkeEGA, Gpredarmax2


def cm2inch(value):
    return value/2.54

import matplotlib as mpl

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
#mpl.rcParams['text.latex.preamble'] = '\usepackage[T1]{fontenc}, \usepackage{txfonts}'

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


#plt.figure()
#plt.subplot(311)
#plt.plot(tt/60/24,G,'k')
#plt.title(r'Glycemia')
#plt.xlabel(r'time [days]')
#plt.ylabel(r'$G$ [mmol/L]')
#plt.subplot(312)
#plt.plot(tt/60/24,vsig/1000,'k')
#plt.title(r'rate of insulin administration')
#plt.yticks([0,0.4,0.8,1.2])
#plt.xlabel(r'čas [days]')
#plt.ylabel(r'$v$ [U/min]')
#plt.subplot(313)
#plt.plot(tt/60/24,dsig,'k')
#plt.title(r'rate of carbohydrate intake')
#plt.yticks([0,4,8,12,16])
#plt.xlabel(r'time [days]')
#plt.ylabel(r'$d$ [g/min]')
#plt.savefig('CGMdata.pdf')
#plt.close('all')


Gpb=8
fz=1
N=12
mao=0

#GpredN,theta,thetak,Gf=Gpredarx2(G,Gpb,dsig,vsig,6,6,6,N,fz,mao)
GpredN,theta,thetak,Gf=Gpredarmax2(G,Gpb,dsig,vsig,6,6,6,6,N,fz,mao)

Cdata=dsig*Ts
Vbasdata=vbas/1e3*60
Vboldata=vbol/1e3*Ts

plt.figure()
plt.subplot(311)
plt.ylim([0,30])
plt.xlim([0,5.1])
plt.plot(tt/60/24,G,'ko',markersize=0.8,label='CGM')
plt.plot(tt/60/24,GpredN,'gray',label=r'prediction')
plt.title(r'Glycemia')
plt.xlabel(r'time [days]')
plt.ylabel(r'$G$ [mmol/L]')
plt.legend()
plt.subplot(312)
plt.title(r'Carbohydrates')
plt.xlim([0,5.1])
plt.stem(tt[Cdata>0]/60/24,Cdata[Cdata>0],'k',markerfmt='ko', basefmt='k')
plt.xlabel(r'time [days]')
plt.ylabel('C [g]')

plt.subplot(313)
plt.title(r'Basal (full line) and bolus (stem) insulin')
plt.xlim([0,5.1])
plt.plot(tt/60/24,Vbasdata,'k')
plt.xlabel(r'time [days]')
plt.ylabel(r'Basal [U/h]')
plt.twinx()
plt.xlim([0,5.1])
plt.stem(tt[Vboldata>0]/60/24,Vboldata[Vboldata>0],'k',markerfmt='ko', basefmt='k')
plt.xlabel(r'time [days]')
plt.ylabel(r'Bolus [U]')
plt.savefig('Glycemia prediction.pdf')
plt.close('all')


#plt.subplot(212)
#plt.plot(tt/60/24,theta[:,0],'k-')
#plt.plot(tt/60/24,theta[:,1:],'k-')
#plt.title(r'model parameters')
#plt.xlabel(r'čas [dni]')
#plt.ylabel(r'$\theta$')
plt.savefig('prediction.pdf')
plt.close('all')

percentages=clarkeEGA(G,GpredN,288,N,'EGA')
