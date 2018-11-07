# functions for simulation of Bergman minimal model with three different models of insulin secretion

import numpy as np
from scipy.integrate import odeint

# --- model equations
# the ,,S'' signal means insulin secretion in all three cases
# this signal has also lower saturation to prevent the negative secretion rate

# minimal model with only proportional (P) secretion model
def fcn_BergmanGI1(x,t,p,RaG,RaI,Gb,Ib):      # RaG [mmol/kg/min] , RaI [mU/kg/min]
    G, X, I = x
    Tg, Kx, V_G, Tx, Ti, Kg1, V_I = p           
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * RaG   
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)  
    
    S = Kg1*(G-Gb)
    I_dot= -1/Ti* (I-Ib) + np.max([S,0]) + 1/V_I * RaI
    
    return np.array([G_dot, X_dot, I_dot])

# minimal model with proportional-derivative (PD) secretion model
def fcn_BergmanGI2(x,t,p,RaG,RaI,Gb,Ib):      # RaG [mmol/kg/min] , RaI [mU/kg/min]
    G, X, I, v2 = x
    Tg, Kx, V_G, Tx, Ti, Kg1, Kg2, T2, V_I = p           
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * RaG   
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)  
    
    S = Kg1*(G-Gb) + Kg2/T2*(G-Gb)-v2
    I_dot= -1/Ti* (I-Ib) + np.max([S,0]) + 1/V_I * RaI
    v2_dot= -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
    
    return np.array([G_dot, X_dot, I_dot, v2_dot])

# minimal model with proportional-derivative (PD) secretion model
# proportional part here is filtered through 1st order filter
def fcn_BergmanGI3(x,t,p,RaG,RaI,Gb,Ib):      # RaG [mmol/kg/min] , RaI [mU/kg/min]
    G, X, I, v1, v2 = x
    Tg, Kx, V_G, Tx, Ti, Kg1, T1, Kg2, T2, V_I = p           
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * RaG   
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)  
    
    S = v1 + Kg2/T2*(G-Gb)-v2
    I_dot= -1/Ti* (I-Ib) + np.max([S,0]) + 1/V_I * RaI
    v1_dot= -1/T1 * v1 + Kg1/T1 * (G-Gb)
    v2_dot= -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
    
    return np.array([G_dot, X_dot, I_dot, v1_dot, v2_dot])


# ---- simulation functions
    
def sim_BergmanGI1(t,p,RaG,RaI,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, Ib])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGI1,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       RaG[i-1],RaI[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
    return x

def sim_BergmanGI2(t,p,RaG,RaI,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, Ib, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGI2,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       RaG[i-1],RaI[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
    return x

def sim_BergmanGI3(t,p,RaG,RaI,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, Ib, 0, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGI3,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       RaG[i-1],RaI[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
    return x


# functions for simulation of hyperinsulinemic euglycemic glucose clamp
    
def sim_BergmanGI1_clamp(t,p,RaI,Gb,Ib,BW):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, Ib])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    RaG=np.zeros(idx_final)
    RaG[0]=280/180/BW
    Ts2=5
    k=0
    r=0
    gammac=8.87/10
    FM=[0, 1]
    SM=[0, 280/180, 280/180]
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGI1,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       RaG[i-1],RaI[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
        if t[i]>=Ts2*k:           
            Gh=x[i,0]
            if k==0:
                r=280/180
            elif k==1:
                r=280/180
            else:
                FM[0]=Gb/Gh
                SM[0]=SM[2]*FM[1]*FM[0]
                r=gammac*(Gb-Gh)+SM[0]
                FM=np.roll(FM,1)
                SM=np.roll(SM,1)  
            k+=1  
        RaG[i]=r/BW
    return x,RaG

def sim_BergmanGI2_clamp(t,p,RaI,Gb,Ib,BW):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, Ib, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    RaG=np.zeros(idx_final)
    RaG[0]=280/180/BW
    Ts2=5
    k=0
    r=0
    gammac=8.87/10
    FM=[0, 1]
    SM=[0, 280/180, 280/180]
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGI2,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       RaG[i-1],RaI[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
        if t[i]>=Ts2*k:           
            Gh=x[i,0]
            if k==0:
                r=280/180
            elif k==1:
                r=280/180
            else:
                FM[0]=Gb/Gh
                SM[0]=SM[2]*FM[1]*FM[0]
                r=gammac*(Gb-Gh)+SM[0]
                FM=np.roll(FM,1)
                SM=np.roll(SM,1)  
            k+=1  
        RaG[i]=r/BW
    return x,RaG

def sim_BergmanGI3_clamp(t,p,RaI,Gb,Ib,BW):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, Ib, 0, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    RaG=np.zeros(idx_final)
    RaG[0]=280/180/BW
    Ts2=5
    k=0
    r=0
    gammac=8.87/10
    FM=[0, 1]
    SM=[0, 280/180, 280/180]
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGI3,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       RaG[i-1],RaI[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
        if t[i]>=Ts2*k:           
            Gh=x[i,0]
            if k==0:
                r=280/180
            elif k==1:
                r=280/180
            else:
                FM[0]=Gb/Gh
                SM[0]=SM[2]*FM[1]*FM[0]
                r=gammac*(Gb-Gh)+SM[0]
                FM=np.roll(FM,1)
                SM=np.roll(SM,1)  
            k+=1  
        RaG[i]=r/BW
    return x,RaG


