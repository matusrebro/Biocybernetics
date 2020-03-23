# functions for simulation of Bergman minimal model with three different models of insulin secretion

import numpy as np
from scipy.integrate import odeint

# --- model equations
# the ,,S'' signal means insulin secretion in all three cases
# this signal has also lower saturation to prevent the negative secretion rate

# --- intravenous inputs (glucose+insulin)

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

# --- oral glucose input (incretin effect)
# d is the glucose/carbohydrate rate of intake [mmol/kg/min]

# in this model the incretin effect is modeled by multiplicative modification of pancreas parameters
def fcn_BergmanGIo0(x,t,p,d,Gb,Ib):      
    G, X, D, Ra, I, v2 = x
    Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, Kg1m, Kg2m = p           
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * Ra   
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)   
    D_dot= -1/Td1 * D + Kd/Td1 * d
    Ra_dot= -1/Td2 * Ra + 1/Td2 * D
    
    S = Kg1m*Kg1*(G-Gb) + Kg2m*Kg2/T2*(G-Gb)-v2
    I_dot= -1/Ti* (I-Ib) + np.max([S,0]) 
    v2_dot= -1/T2 * v2 + Kg2m*Kg2/(T2**2) * (G-Gb)
    
    return np.array([G_dot, X_dot, D_dot, Ra_dot, I_dot, v2_dot])

# here the incretin effect is modeled via signals from glucose absorption submodel
def fcn_BergmanGIo1(x,t,p,d,Gb,Ib):
    G, X, D, Ra, I, v2 = x
    Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, Kg3a, Kg3b = p           
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * Ra   
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)   
    D_dot= -1/Td1 * D + Kd/Td1 * d
    Ra_dot= -1/Td2 * Ra + 1/Td2 * D
    
    S = Kg1*(G-Gb) + Kg2/T2*(G-Gb)-v2 + Kg3a*D + Kg3b*Ra
    I_dot= -1/Ti* (I-Ib) + np.max([S,0]) 
    v2_dot= -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
    
    return np.array([G_dot, X_dot, D_dot, Ra_dot, I_dot, v2_dot])

# same as before, but 1st order dynamics is added
def fcn_BergmanGIo2(x,t,p,d,Gb,Ib):
    G, X, D, Ra, I, v2, v3 = x
    Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, T3, Kg3a, Kg3b = p           
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * Ra   
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)   
    D_dot= -1/Td1 * D + Kd/Td1 * d
    Ra_dot= -1/Td2 * Ra + 1/Td2 * D
    
    S = Kg1*(G-Gb) + Kg2/T2*(G-Gb) - v2 + v3
    I_dot= -1/Ti* (I-Ib) + np.max([S,0]) 
    v2_dot= -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
    v3_dot= -1/T3 * v3 + Kg3a*D/T3 + Kg3b*Ra/T3
    
    return np.array([G_dot, X_dot, D_dot, Ra_dot, I_dot, v2_dot, v3_dot])

# this is combination of the first model and the model before
# multiplicative modification of pancreas parameters is dependent on glucose absorption signals and thus the effect is nonlinear
def fcn_BergmanGIo3(x,t,p,d,Gb,Ib):      # Ra [mmol/min] , I [mU/L]
    G, X, D, Ra, I, v2, v3 = x
    Tg, Kx, V_G, Tx, Kd, Td1, Td2, Ti, Kg1, Kg2, T2, Kg1m, Kg2m, T3, Kg3a, Kg3b = p           
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * Ra   
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)   
    D_dot= -1/Td1 * D + Kd/Td1 * d
    Ra_dot= -1/Td2 * Ra + 1/Td2 * D
    
    S = (1+Kg1m*v3)*Kg1*(G-Gb) + (1+Kg2m*v3)*Kg2/T2*(G-Gb)-v2
    I_dot= -1/Ti* (I-Ib) + np.max([S,0])
    v2_dot= -1/T2 * v2 + (1+Kg2m*v3)*Kg2/(T2**2) * (G-Gb)
    v3_dot= -1/T3 * v3 + Kg3a*D/T3 + Kg3b*Ra/T3
    
    return np.array([G_dot, X_dot, D_dot, Ra_dot, I_dot, v2_dot, v3_dot])


# ---- exercise effects
# model of exercise effect on glucose and insulin levels

def fcn_Ex(x,t,p,ux,Gb,Ib):
    G, X, I, v2, PVO, Gprod, Gup, Ggly, Ie, A = x
    Tg, Kx, V_G, Tx, Ti, Kg1, Kg2, T2, V_I, a1, a2, a3, a4, a5, a6, k, Tgly  = p   
    
    G_dot= -Kx * X*G - 1/Tg * (G-Gb) + 1/V_G * (Gprod - Gup - Ggly)
    X_dot= -1/Tx * X + 1/Tx * (I-Ib)
    S = Kg1*(G-Gb) + Kg2/T2*(G-Gb)-v2
    I_dot= -1/Ti* (I-Ib) + np.max([S,0]) - Ie
    v2_dot= -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)

    PVO_dot = -0.8*PVO + 0.8*ux
    Gprod_dot = -a2*Gprod + a1*PVO    
    Gup_dot = -a4*Gup + a3*PVO  
    
    Ath = -1.1521*ux**2 + 87.47*ux
    
    Ggly_dot = 0
    if A<Ath:
        pass
    elif A>=Ath:
        Ggly_dot=k
        
    if ux==0:
        Ggly_dot= - Ggly/Tgly
 
    Ie_dot = -a6*Ie + a5*PVO
       
    A_dot = ux
    if ux>0:
        pass
    elif ux==0:
        A_dot = -A/0.001

    return np.array([G_dot, X_dot, I_dot, v2_dot, PVO_dot, Gprod_dot, Gup_dot, Ggly_dot, Ie_dot, A_dot])


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

# simulation functions for oral glucose inputs
def sim_BergmanGIo0(t,p,d,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, 0, 0, Ib, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGIo0,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       d[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
    return x

def sim_BergmanGIo1(t,p,d,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, 0, 0, Ib, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGIo1,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       d[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
    return x

def sim_BergmanGIo2(t,p,d,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, 0, 0, Ib, 0, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGIo2,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       d[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
    return x

def sim_BergmanGIo3(t,p,d,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, 0, 0, Ib, 0, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_BergmanGIo3,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       d[i-1],Gb,Ib,)
                 )
        x[i,:] = y[-1,:]
    return x

def sim_Ex(t,p,ux,Gb,Ib):
    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1    
    x0=np.array([Gb, 0, Ib, 0, 0, 0, 0, 0, 0, 0])   
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0   
    for i in range(1,idx_final):
        y=odeint(fcn_Ex,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       ux[i-1],Gb, Ib, )
                 )
        x[i,:] = y[-1,:]        
    return x