import numpy as np
from scipy.integrate import odeint
from copy import deepcopy

# Sorensen model differential equations
# x - state vector
# t - time vector
# p - model parameters (only some of the pancreas parameters)
# Basal - basal (steady state) values of states and some signals
# r_IVG - intravenous glucose infusion rate [mmol/kg/min] 
# r_IVI - intravenous insulin infusion rate [mU/kg/min] 

def fcn_Sorensen(x,t,p,Basal,r_IVG,r_IVI):
    # state signals
    G_BV, G_BI, G_H, G_G, G_L, G_K, G_PV, G_PI, MI_HGP, ff, MI_HGU, I_B, I_H, I_G, I_L, I_K, I_PV, I_PI, P, I, Q, Gamma = x#0    
    # basal state values
    GB_BV, GB_BI, GB_H, GB_G, GB_L, GB_K, GB_PV, GB_PI, IB_B, IB_H, IB_G, IB_L, IB_K, IB_PV, IB_PI, GammaB, rB_PIR, SB = Basal
    # model parameters
    alpha, beta, K, M1, M2, gamma = p 
    
    # whole blood water glucose distribution volume of avg. man
    VG = 5.6/70*0.8 # [L/kg]
    # total interstitial volume
    Vi1 = 8.4/70 # [L/kg]
    # total intracellular volume
    Vi2 = 22.05/70 # [L/kg]
    
    # division of this volume between organs
    # [L/kg]
    VG_BV = VG*0.073  
    VG_H  = VG*0.293
    VG_L  = VG*0.161 + Vi1*0.071 + Vi2*0.052
    VG_G  = VG*0.127 + Vi1*0.062
    VG_K  = VG*0.121 + Vi1*0.011
    VG_PV = VG*0.225
    
    # interstitial volumes of brain and peripheries
    # [L/kg]
    VG_BI = Vi1*0.054
    VG_PI = Vi1*0.802
  
    # total flow rate of whole blood water glucose
    # [L/min/kg]
    QG = 5.2/70*0.8
    
    # flow rates of individual organs
    # [L/min/kg]
    QG_B = QG*0.135
    QG_H = deepcopy(QG)
    QG_A = QG*0.058
    QG_L = QG*0.288
    QG_G = QG*0.231
    QG_K = QG*0.231
    QG_P = QG*0.346
    
    # transcapillary diffusion time constants 
    # [min]
    TG_B = 2.1
    TG_P = 5

    # plasma insulin distribution volume of avg. man
    VI = 5.6/70*0.6 # [L/kg]

    # [L/kg]
    VI_B = VI*0.073
    VI_H = VI*0.293
    VI_L = VI*0.161 + Vi1*0.071
    VI_G = VI*0.127 + Vi1*0.062
    VI_K = VI*0.121 + Vi1*0.011
    VI_PV= VI*0.225
    
    # interstitial volume of peripheries
    # [L/kg]
    VI_PI = Vi1*0.802

    # total flow rate of plasma insulin
    # [L/min/kg]
    QI = 5.2/70*0.6
    
    # [L/min/kg]
    QI_B = QI*0.135
    QI_H = deepcopy(QI)
    QI_A = QI*0.058
    QI_L = QI*0.288
    QI_G = QI*0.231
    QI_K = QI*0.231
    QI_P = QI*0.346
    
    # transcapillary diffusion time constant  
    # [min]
    TI_P = 20
    
    # plasma glucagon distribution volume
    # [L/kg]
    V_Ga = VI_B + VI_H + VI_G + VI_L + VI_K + VI_PV + VI_PI
    
    # labile insulin amount extrapolated for zero glucose concentration
    # [U]
    Q0   = 6.33
    
    # here goes the Glucose part
    
    # ---- metabolic rates
    # [mmol/min/kg]
    
    # basal hepatic glucose production
    rB_HGP = 0.861/70
    
    # constant (brain, RBCs, gut)
    # [mmol/min/kg]
    r_BGU  = rB_HGP*0.451 # brain glucose uptake
    r_RBCU = rB_HGP*0.065 # red blood cell glucose uptake
    r_GGU  = rB_HGP*0.129 # gut glucose uptake
    
    # basal metabolic rates (hepatic and peripheral uptakes)
    # [mmol/min/kg]
    rB_HGU = rB_HGP*0.129 # hepatic glucose uptake
    rB_PGU = rB_HGP*0.226 # peripheral glucose uptake
    
    # metabolic multipliers
    a1=6.52
    b1=0.338
    c1=5.82
    MI_PGU= (1-a1*np.tanh(b1 * (1 - c1))) + a1*np.tanh(b1* (I_PI/IB_PI - c1))    
    MG_PGU= G_PI/GB_PI
    r_PGU = MI_PGU * MG_PGU * rB_PGU

    a2=1.41
    b2=0.62
    c2=0.497
    MG_HGP = (1+a2*np.tanh(b2 * (1 - c2)) ) - a2*np.tanh(b2 * (G_L/GB_L - c2))  
    MGa0_HGP = 1/(np.tanh(0.39*1))*np.tanh(0.39*Gamma/GammaB)
    MGa_HGP = MGa0_HGP - ff
    r_HGP = MI_HGP * MG_HGP * MGa_HGP * rB_HGP   

    a3=1.14
    b3=1.66
    c3=0.89
    MIinf_HGP = (1+a3*np.tanh(b3 * (1 - c3))) - a3*np.tanh(b3 * (I_L/IB_L - c3))

    a4=5.66
    b4=2.44
    c4=1.48
    MG_HGU= (1-a4*np.tanh(b4 * (1 - c4)) ) + a4*np.tanh(b4 * (G_L/GB_L - c4)) 
    r_HGU = MI_HGU * MG_HGU * rB_HGU

    a5=0.55
    MIinf_HGU = 1/(np.tanh(a5 * 1) )*np.tanh(a5 * I_L/IB_L)
    
    # kidney glucose extraction
    r_KGE = 0
    if G_K>=0 and G_K<25.56:       
        r_KGE = 0.00563 + 0.00563*np.tanh(0.198* (G_K - 25.56)) 
    elif G_K>=25.56:
        r_KGE = -0.0262 + 0.00125 * G_K
    # ---- dif. eq. for glycemia
    dot_G_BV = 1/VG_BV * (QG_B*(G_H - G_BV) - VG_BI/TG_B*(G_BV - G_BI))
    dot_G_BI = 1/VG_BI * (VG_BI/TG_B*(G_BV - G_BI) - r_BGU)
    dot_G_H  = 1/VG_H  * (QG_B*G_BV + QG_L*G_L + QG_K*G_K + QG_P*G_PV - QG_H*G_H - r_RBCU + r_IVG)
    dot_G_G  = 1/VG_G  * (QG_G*(G_H - G_G) - r_GGU)
    dot_G_L  = 1/VG_L  * (QG_A*G_H + QG_G*G_G - QG_L*G_L + r_HGP - r_HGU)
    dot_G_K  = 1/VG_K  * (QG_K*(G_H - G_K) - r_KGE)
    dot_G_PV = 1/VG_PV * (QG_P*(G_H - G_PV) - VG_PI/TG_P*(G_PV - G_PI))
    dot_G_PI = 1/VG_PI * (VG_PI/TG_P * (G_PV - G_PI) - r_PGU)
    
    #---- dif. eq. for metabolic rates
    dot_MI_HGP = 1/25 * (MIinf_HGP - MI_HGP)
    dot_ff     = 1/65 * ((MGa0_HGP - 1)/2 - ff)
    dot_MI_HGU = 1/25 * (MIinf_HGU - MI_HGU)
    
    # here goes the Insulin part
     
    # Pancreas stuff:
    X = G_H**3.12 / (7.85**3.12 + 3.06*G_H**2.88)
    P_inf = X**1.1141 
    Y = X**1.1141 
    
    S = Q * (M1 * Y + M2 * np.max([(X-I),0]))  
     
    #---- dif. eq. for pancreas
    dot_P = alpha * (P_inf - P)
    dot_I = beta  * (X - I)
    dot_Q = K * (Q0 - Q) + gamma * P - S

    # metabolic rates
    # [mU/min/kg]
    r_PIR = S/SB * rB_PIR # pancreatic insulin release
    r_LIC = 0.4 * (QI_A * I_H + QI_G * I_G + r_PIR)  # liver insulin clearance
    r_KIC = 0.3 * QI_K * I_H # kidney insulin clearance
    r_PIC = I_PI / ((1-0.15) / (0.15 * QI_P) - TI_P/VI_PI) # peripheral insulin clearance
    
    #---- dif. eq. for insulin
    dot_I_B = 1/VI_B  * (QI_B * (I_H - I_B))
    dot_I_H = 1/VI_H  * (QI_B * I_B + QI_L * I_L + QI_K * I_K + QI_P * I_PV - QI_H * I_H + r_IVI)
    dot_I_G = 1/VI_G  * (QI_G * (I_H - I_G))    
    dot_I_L = 1/VI_L  * (QI_A * I_H + QI_G * I_G - QI_L * I_L + r_PIR - r_LIC)
    dot_I_K = 1/VI_K  * (QI_K * (I_H - I_K) - r_KIC)    
    dot_I_PV= 1/VI_PV * (QI_P * (I_H - I_PV) - VI_PI/TI_P * (I_PV - I_PI))
    dot_I_PI= 1/VI_PI * (VI_PI/TI_P * (I_PV - I_PI) - r_PIC)
       
    # here goes the Glucagon part

    r_PGaC  = 0.91 * Gamma/70
    rB_PGaR = 0.91/70
    MG_PGaR = 1+2.10*np.tanh(4.18 * (1 - 0.62)) - 2.10*np.tanh(4.18 * (G_H/GB_H - 0.62))
    MI_PGaR = 1+0.61*np.tanh(1.06 * (1 - 0.47)) - 0.61*np.tanh(1.06 * (I_H/IB_H - 0.47)) 
    
    r_PGaR = MG_PGaR * MI_PGaR * rB_PGaR

    dot_Gamma = 1/V_Ga * (r_PGaR - r_PGaC)
        #                G_BV,     G_BI,     G_H,     G_G,     G_L,     G_K,     G_PV,     G_PI,     MI_HGP,     ff,     MI_HGU,     I_B,     I_H,     I_G,     I_L,     I_K,     I_PV,     I_PI,     P,     I,     Q,    Gamma
    return np.array([dot_G_BV, dot_G_BI, dot_G_H, dot_G_G, dot_G_L, dot_G_K, dot_G_PV, dot_G_PI, dot_MI_HGP, dot_ff, dot_MI_HGU, dot_I_B, dot_I_H, dot_I_G, dot_I_L, dot_I_K, dot_I_PV, dot_I_PI, dot_P, dot_I, dot_Q, dot_Gamma])


# this function uses ode solver to simulate Sorensen model
# it returns array of all model states
def sim_Sorensen(t,p,Gb,Ib,r_IVG,r_IVI):

    alpha, beta, K, M1, M2, gamma = p   

    # total interstitial volume
    Vi1 = 8.4/70 # [L/kg]

    # interstitial volumes of brain and peripheries
    # [L/kg]
    VG_BI = Vi1*0.054
    VG_PI = Vi1*0.802

    # total flow rate of whole blood water glucose
    # [L/min/kg]
    QG = 5.2/70*0.8
    
    QG_B = QG*0.135
    QG_A = QG*0.058
    QG_L = QG*0.288
    QG_G = QG*0.231
    QG_P = QG*0.346
    
    TG_B = 2.1
    TG_P = 5

    # interstitial volume of peripheries
    # [L/kg]
    VI_PI = Vi1*0.802

    # total flow rate of plasma insulin
    # [L/min/kg]
    QI = 5.2/70*0.6
    
    # [L/min/kg]
    QI_B = QI*0.135
    QI_H = deepcopy(QI)
    QI_A = QI*0.058
    QI_L = QI*0.288
    QI_G = QI*0.231
    QI_K = QI*0.231
    QI_P = QI*0.346
    
    TI_P = 20
        
    Q0   = 6.33

    # steady state stuff
    
    # basal hepatic glucose production
    rB_HGP = 0.861/70    
    
    # constant (brain, RBCs, gut)
    # [mmol/min/kg]
    r_BGU  = rB_HGP*0.451
    r_GGU  = rB_HGP*0.129
    
    # basal metabolic rates (hepatic and peripheral uptakes)
    # [mmol/min/kg]
    rB_HGU = rB_HGP*0.129
    rB_PGU = rB_HGP*0.226
        
    
    # here goes the steady state calculation
    # most of the time the peripheral concentrations are measured
    # ...but if you simulate, for example some glucose clamp where arterial
    # measurements are done, comment/uncomment lines where needed
    
    GB_PV = Gb # use this if peripheral glucose is known
#    GB_H = Gb # use this if arterial glucose is known
    
    GB_H  = rB_PGU/QG_P + GB_PV # use this if peripheral glucose is known
#    GB_PV = GB_H - rB_PGU/QG_P # use this if arterial glucose is known
    GB_PI = -TG_P/VG_PI*rB_PGU + GB_PV 
    GB_H  = rB_PGU/QG_P + GB_PV
    GB_K  = GB_H
    GB_G  = -r_GGU/QG_G + GB_H
    GB_L  = 1/QG_L * (QG_A * GB_H + QG_G * GB_G + rB_HGP - rB_HGU) 
    GB_BV = -r_BGU/QG_B + GB_H
    GB_BI = -r_BGU * TG_B/VG_BI + GB_BV
       
    IB_PV = Ib # use this if peripheral insulin is known
#    IB_H = Ib # use this if arterial insulin is known
    
    IB_H = IB_PV/(1-0.15) # use this if peripheral insulin is known
#    IB_PV= IB_H*(1-0.15) # use this if arterial insulin is known
    IB_K = IB_H*(1-0.3)
    IB_B = IB_H
    IB_G = IB_H
    IB_PI= IB_PV - (QI_P*TI_P/VI_PI * (IB_H - IB_PV)) 
    IB_L = 1/QI_L * (QI_H * IB_H - QI_B * IB_B - QI_K * IB_K - QI_P * IB_PV)

    rB_PIR = QI_L/(1-0.4) * IB_L - QI_G * IB_G - QI_A * IB_H
    
    XB = GB_H**3.12 / (7.85**3.12 + 3.06*GB_H**2.88)
    PB_inf = XB**1.1141 
    YB = XB**1.1141 
    IB = deepcopy(XB)
    PB = deepcopy(PB_inf)
    
    QB = (K*Q0 + gamma*PB)/(K + M1*YB)
    SB = M1 * YB * QB
    
    GammaB = 1
    
    MIB_HGP = 1
    MIB_HGU = 1
    ffB = 0
    x0 = [GB_BV, GB_BI, GB_H, GB_G, GB_L, GB_K, GB_PV, GB_PI, MIB_HGP, ffB, MIB_HGU, IB_B, IB_H, IB_G, IB_L, IB_K, IB_PV, IB_PI, PB, IB, QB, GammaB]
    Basal = [GB_BV, GB_BI, GB_H, GB_G, GB_L, GB_K, GB_PV, GB_PI, IB_B, IB_H, IB_G, IB_L, IB_K, IB_PV, IB_PI, GammaB, rB_PIR, SB]

    Ts=t[1]-t[0]
    idx_final=t.size
    
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0

    for i in range(1,idx_final):
        y=odeint(fcn_Sorensen,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       Basal,
                       r_IVG[i-1],r_IVI[i-1],), rtol=1e-5
                 )
        x[i,:] = y[-1,:]    
    
    return x

# this function uses ode solver to simulate euglycemic glucose clamp for Sorensen model
# it returns state vector and glucose infusion vector (r_IVG [mmol/min/kg])
# glucose infusion signal is generated in simulation
# additional function argument - BW [kg] is bodyweight of simulated subject
def sim_Sorensen_eugc(t,p,Gb,Ib,r_IVI,BW):

    alpha, beta, K, M1, M2, gamma = p 

    # total interstitial volume
    Vi1 = 8.4/70 # [L/kg]

    # interstitial volumes of brain and peripheries
    # [L/kg]
    VG_BI = Vi1*0.054
    VG_PI = Vi1*0.802

    # total flow rate of whole blood water glucose
    # [L/min/kg]
    QG = 5.2/70*0.8
    
    QG_B = QG*0.135
    QG_A = QG*0.058
    QG_L = QG*0.288
    QG_G = QG*0.231
    QG_P = QG*0.346
    
    TG_B = 2.1
    TG_P = 5

    # interstitial volume of peripheries
    # [L/kg]
    VI_PI = Vi1*0.802

    # total flow rate of plasma insulin
    # [L/min/kg]
    QI = 5.2/70*0.6
    
    # [L/min/kg]
    QI_B = QI*0.135
    QI_H = deepcopy(QI)
    QI_A = QI*0.058
    QI_L = QI*0.288
    QI_G = QI*0.231
    QI_K = QI*0.231
    QI_P = QI*0.346
    
    TI_P = 20
        
    Q0   = 6.33

    # steady state stuff
    
    # basal hepatic glucose production
    rB_HGP = 0.861/70    
    
    # constant (brain, RBCs, gut)
    # [mmol/min/kg]
    r_BGU  = rB_HGP*0.451
    r_GGU  = rB_HGP*0.129
    
    # basal metabolic rates (hepatic and peripheral uptakes)
    # [mmol/min/kg]
    rB_HGU = rB_HGP*0.129
    rB_PGU = rB_HGP*0.226
        
    # here goes the steady state calculation
    # in this case we assume that glucose and insulin measurements are done from
    # arterial blood, since we are doing the glucose clamp experiment
    # comment/uncomment lines where needed
    
#    GB_PV = Gb # use this if peripheral glucose is known
    GB_H = Gb # use this if arterial glucose is known
    
#    GB_H  = rB_PGU/QG_P + GB_PV # use this if peripheral glucose is known
    GB_PV = GB_H - rB_PGU/QG_P # use this if arterial glucose is known
    GB_PI = -TG_P/VG_PI*rB_PGU + GB_PV 
    GB_H  = rB_PGU/QG_P + GB_PV
    GB_K  = GB_H
    GB_G  = -r_GGU/QG_G + GB_H
    GB_L  = 1/QG_L * (QG_A * GB_H + QG_G * GB_G + rB_HGP - rB_HGU) 
    GB_BV = -r_BGU/QG_B + GB_H
    GB_BI = -r_BGU * TG_B/VG_BI + GB_BV

   
#    IB_PV = Ib # use this if peripheral insulin is known
    IB_H = Ib # use this if arterial insulin is known
    
#    IB_H = IB_PV/(1-0.15) # use this if peripheral insulin is known
    IB_PV= IB_H*(1-0.15) # use this if arterial insulin is known
    IB_K = IB_H*(1-0.3)
    IB_B = IB_H
    IB_G = IB_H
    IB_PI= IB_PV - (QI_P*TI_P/VI_PI * (IB_H - IB_PV)) 
    IB_L = 1/QI_L * (QI_H * IB_H - QI_B * IB_B - QI_K * IB_K - QI_P * IB_PV)

    rB_PIR = QI_L/(1-0.4) * IB_L - QI_G * IB_G - QI_A * IB_H
    
    XB = GB_H**3.12 / (7.85**3.12 + 3.06*GB_H**2.88)
    PB_inf = XB**1.1141 
    YB = XB**1.1141 
    IB = deepcopy(XB)
    PB = deepcopy(PB_inf)
    
    QB = (K*Q0 + gamma*PB)/(K + M1*YB)
    SB = M1 * YB * QB
    
    GammaB = 1
    
    MIB_HGP = 1
    MIB_HGU = 1
    ffB = 0
    x0 = [GB_BV, GB_BI, GB_H, GB_G, GB_L, GB_K, GB_PV, GB_PI, MIB_HGP, ffB, MIB_HGU, IB_B, IB_H, IB_G, IB_L, IB_K, IB_PV, IB_PI, PB, IB, QB, GammaB]
    Basal = [GB_BV, GB_BI, GB_H, GB_G, GB_L, GB_K, GB_PV, GB_PI, IB_B, IB_H, IB_G, IB_L, IB_K, IB_PV, IB_PI, GammaB, rB_PIR, SB]

    Ts=t[1]-t[0]
    idx_final=t.size
    
    x=np.zeros([idx_final,len(x0)])
    x[0,:]=x0
    r_IVG=np.zeros(idx_final)
    # here are the parameters and initial conditions for euglycemic clamp technique 
    # feedback algorithm 
    Ts2=5 # sample time for glucose measurements [min]
    k=0
    r=0
    gammac=8.87/10
    FM=[0, 1]
    SM=[0, 280/180, 280/180]

    for i in range(1,idx_final):
        y=odeint(fcn_Sorensen,x[i-1,:],np.linspace((i-1)*Ts,i*Ts),
                 args=(p,
                       Basal,
                       r_IVG[i-1],r_IVI[i-1],), rtol=1e-5
                 )
        x[i,:] = y[-1,:]   
        # here is the algorithm itself
        # there can be many types of euglycemic clamp algorithms
        # this one is implemented on the basis of Sorensen's work
        if t[i]>=Ts2*k:           
            Gh=x[i,2]
            if k==0:
                r=0
            elif k==1:
                r=140/180
            else:
                FM[0]=Gb/Gh
                SM[0]=SM[2]*FM[1]*FM[0]
                r=gammac*(Gb-Gh)+SM[0]
                FM=np.roll(FM,1)
                SM=np.roll(SM,1)  
            k+=1  
        r_IVG[i]=r/BW
    
    return x,r_IVG