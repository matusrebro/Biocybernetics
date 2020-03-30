


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from copy import deepcopy

pathToParameters = 'Model Parameters/Diabetes/'

# general form of minimal (Bergman) model
def fcn_Bergman(x, t, p, RaG, RaI, S, Gb, Ib):      
    
    """
    Input signals:
    
    RaG [mmol/kg/min] - rate of appearance of glucose in blood/plasma
    RaI [mU/kg/min] - rate of appearance of insulin in plasma
    S   [mU/L/min] - insulin secretion bz pancreas
    
    Parameters:
    
    p = [Tg, Kx, V_G, Tx, Ti, V_I]
    
    T_G [min] - time constant of glucose compartment (inverse of so called glucose effectiveness)
    Kx [L/mU/min] - insulin sensitivity index
    V_G [L/kg] - glucose distribution volume
    T_X [min] - time constant of X (remote insulin) compartment
    T_I [min] - time constant of insulin compartment
    V_I [min] - insulin distribution volume
    
    Basal (steady state) values:
        
    Gb [mmol/L] basal glucose concentration
    Ib [mU/L] basal insulin concentration
    
    """
    
    G, X, I = x
    T_G, Kx, V_G, T_X, T_I, V_I = p           
    G_dot= -Kx * X*G - 1/T_G * (G-Gb) + 1/V_G * RaG   
    X_dot= -1/T_X * X + 1/T_X * (I-Ib)  
    I_dot= -1/T_I* (I-Ib) + np.max([S,0]) + 1/V_I * RaI
    
    return np.array([G_dot, X_dot, I_dot])

# general model structure for both oral and intravenous inputs
def fcn_Bergman_extended(x, t, p, RaG_iv, d, RaI, Gb, Ib,
                         incretin_effect_model = 'model1'):
    
    G, X, D, Ra, I, v2, v3 = x    
    
    if incretin_effect_model == 'model1':
        
        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, Kg1, Kg2, T2, V_I, Kg1m, Kg2m = p  
        
        
        S = Kg1m*Kg1*(G-Gb) + Kg2m*Kg2/T2*(G-Gb)-v2
        v2_dot = -1/T2 * v2 + Kg2m*Kg2/(T2**2) * (G-Gb)
        
        v3_dot = 0
                
    elif incretin_effect_model == 'model2':

        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, Kg1, Kg2, T2, V_I, Kg3a, Kg3b = p
        
        S = Kg1*(G-Gb) + Kg2/T2*(G-Gb)-v2 + Kg3a*D + Kg3b*Ra
        
        v2_dot = -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
        v3_dot = 0

     
    elif incretin_effect_model == 'model3':
        
        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, Kg1, Kg2, T2, T3, V_I, Kg3a, Kg3b = p  
        
        S = Kg1*(G-Gb) + Kg2/T2*(G-Gb) - v2 + v3
        
        v2_dot= -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
        v3_dot= -1/T3 * v3 + Kg3a*D/T3 + Kg3b*Ra/T3
        
    elif incretin_effect_model == 'model4' :
        
        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, Kg1, Kg2, T2, Kg1m, Kg2m, T3, V_I, Kg3a, Kg3b = p 
        
        S = (1+Kg1m*v3)*Kg1*(G-Gb) + (1+Kg2m*v3)*Kg2/T2*(G-Gb)-v2
        
        v2_dot= -1/T2 * v2 + (1+Kg2m*v3)*Kg2/(T2**2) * (G-Gb)
        v3_dot= -1/T3 * v3 + Kg3a*D/T3 + Kg3b*Ra/T3
    
    elif incretin_effect_model == 'none':
        T_G, Kx, V_G, T_X, T_I, Kg1, Kg2, T2, V_I  = p
        
        S = Kg1*(G-Gb) + Kg2/T2*(G-Gb) - v2
        v2_dot = -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
        v3_dot = 0
    else:
        raise ValueError('Invalid value for incretin_effect_model')


    if incretin_effect_model != 'none':
        D_dot= -1/Td1 * D + Kd/Td1 * d
        Ra_dot= -1/Td2 * Ra + 1/Td2 * D
    else:
        D_dot = 0
        Ra_dot = 0
    G_dot, X_dot, I_dot = fcn_Bergman([G, X, I], 0, [T_G, Kx, V_G, T_X, T_I, V_I], (RaG_iv + Ra), RaI, S, Gb, Ib)
    
    return np.array([G_dot, X_dot, D_dot, Ra_dot, I_dot, v2_dot, v3_dot])


class minimal_model:
    
    # --- class for model with intravenous inputs
    class iv:
        
        Gb = 0
        Ib = 0
        parameters = []
        def init_model(self, Gb, Ib, parameters='normal'):
            self.Gb = Gb
            self.Ib = Ib
            
            if isinstance(parameters, str):
                if parameters == 'normal':
                    self.parameters = np.loadtxt(pathToParameters + 'par_normal_iv.csv')
                elif parameters == 'obese':
                    self.parameters = np.loadtxt(pathToParameters + 'par_obese_iv.csv')
                elif parameters == 't2dm':
                    self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_iv.csv')
                else:
                    raise ValueError('Invalid value for default parameters')
            else:
                self.parameters = parameters
    
                
        def __init__(self, Gb, Ib, parameters = 'normal'):
            self.init_model(Gb, Ib, parameters)
        
        def simulation(self, t, RaG_iv, RaI, plot = True):
            """
            t - time array in minutes
            RaG_iv - intravenous glucose infusion array in mmol/min/kg
            RaI - intravenous insulin infusion array in mU/min/kg
            """        
            
            
            Ts = t[1]-t[0] 
            idx_final = int(t[-1]/Ts)+1    
            x0 = np.array([self.Gb, 0, 0, 0, self.Ib, 0, 0])   
            x = np.zeros([idx_final, len(x0)])
            x[0,:] = x0   
            
            for i in range(1,idx_final):
                y = odeint(fcn_Bergman_extended, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                         args=(self.parameters,
                               RaG_iv[i-1], 0, RaI[i-1], self.Gb, self.Ib,
                               'none', )
                         )
                x[i,:] = y[-1,:]
                
            if plot:
                plt.figure()
                plt.subplot(211)
                plt.plot(t, x[:,0])
                plt.xlabel('time [min]')
                plt.ylabel('glycemia [mmol/L]')
                plt.grid()
                plt.subplot(212)
                plt.plot(t, x[:,4])
                plt.xlabel('time [min]')
                plt.ylabel('plasma insulin [mU/L]')
                plt.grid()
                plt.tight_layout()
            return x

        def ivgtt(self, glucose_dose = 0.3, glucose_bolus_min = 2, insulin_dose = 20, insulin_bolus_min = 5, insulin_dosage_time = 20, plot = True ):
            
            """
            glucose_dose - glucose bolus in g/kg
            glucose_bolus_min - time during the glucose bolus is administred in minutes
            insulin_dose - insulin bolus in mU/kg
            insulin_bolus_min - time during the insulin bolus is administred in minutes
            insulin_dosage_time - time at which the insulin bolus is administred in minutes (from start of the ivgtt)
            """
            
            glucose_bolus = glucose_dose*1e3/180 # [mmol/kg]
            
            t = np.arange(0,180,1)
            RaG_iv = np.zeros_like(t, dtype = float)
            RaG_iv[0:int(glucose_bolus_min/1)] = glucose_bolus/1/glucose_bolus_min
            RaI = np.zeros_like(t, dtype = float)
            RaI[int(insulin_dosage_time/1):int(insulin_dosage_time/1)+int(insulin_bolus_min/1)] = insulin_dose/1/insulin_bolus_min
            
            x = self.simulation(t, RaG_iv, RaI, plot)
            
            return x[:,0], x[:,4] # G, I
        
        def hyperinsulinemic_euglycemic_glucose_clamp(self, BW, insulin_rate = 120, plot = True):
            """
            BW - bodyweight in kilograms
            insulin_rate - constant insulin rate in mU/min/m^2
            body_surface_area - body surface area in m^2
            """        
            
            if BW < 100:
                body_surface_area = 1.9
            else:
                body_surface_area = 2.2
            
            t = np.arange(0,250,1)
            Ts = t[1]-t[0] 
            idx_final = int(t[-1]/Ts)+1    
            x0 = np.array([self.Gb, 0, 0, 0, self.Ib, 0, 0])   
            x = np.zeros([idx_final, len(x0)])
            x[0,:] = x0   
            
            # - insulin subsystem model input
            RaI = np.zeros(idx_final, dtype = float)
            RaI[:] = insulin_rate*body_surface_area/BW
            
            # basic euglycemic glucose clamp algorithm variables
            RaG_iv = np.zeros(idx_final, dtype = float)
            RaG_iv[0] = 280/180/BW
            Ts2 = 5
            k = 0
            r = 0
            gammac = 8.87/10
            FM = [0, 1]
            SM = [0, 280/180, 280/180]
            
            for i in range(1,idx_final):
                y = odeint(fcn_Bergman_extended, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                         args=(self.parameters,
                               RaG_iv[i-1], 0, RaI[i-1], self.Gb, self.Ib,
                               'none', )
                         )
                x[i,:] = y[-1,:]            
                if t[i]>=Ts2*k:           
                    Gh = x[i,0]
                    if k == 0:
                        r = 280/180
                    elif k == 1:
                        r = 280/180
                    else:
                        FM[0] = self.Gb/Gh
                        SM[0] = SM[2]*FM[1]*FM[0]
                        r = gammac*(self.Gb-Gh)+SM[0]
                        FM = np.roll(FM,1)
                        SM = np.roll(SM,1)  
                    k+=1  
                RaG_iv[i] = r/BW
            
            if plot:
                plt.figure()
                plt.subplot(311)
                plt.title('glycemia')
                plt.plot(t, x[:,0])
                plt.xlabel('time [min]')
                plt.ylabel('[mmol/L]')
                plt.grid()
                plt.subplot(312)
                plt.title('plasma insulin')
                plt.plot(t, x[:,4])
                plt.xlabel('time [min]')
                plt.ylabel('[mU/L]')
                plt.grid()
                plt.subplot(313)
                plt.title('glucose infusion')
                plt.plot(t, RaG_iv)
                plt.xlabel('time [min]')
                plt.ylabel('[mmol/min/kg]')
                plt.grid()
                plt.tight_layout()
                
            return x[:,0], x[:,4], RaG_iv # G, I, glucose infusion [mmol/min/kg]
    
    # --- class for model with oral inputs
    class oral:
        
        Gb = 0
        Ib = 0
        parameters = []
        incretin_effect_model = ''
        
        def init_model(self, Gb, Ib, parameters='normal', incretin_effect_model = 'model1'):
            
            self.Gb = Gb
            self.Ib = Ib
            
            if isinstance(parameters, str):
                if parameters == 'normal':
                    if incretin_effect_model == 'model1':
                        self.parameters = np.loadtxt(pathToParameters + 'par_normal_oral_inc1.csv')
                    elif incretin_effect_model == 'model2':
                        self.parameters = np.loadtxt(pathToParameters + 'par_normal_oral_inc2.csv')
                    elif incretin_effect_model == 'model3':
                        self.parameters = np.loadtxt(pathToParameters + 'par_normal_oral_inc3.csv')
                    elif incretin_effect_model == 'model4':
                        self.parameters = np.loadtxt(pathToParameters + 'par_normal_oral_inc4.csv')
                    else:
                        raise ValueError('Invalid value for incretin_effect_model')
                        
                elif parameters == 't2dm':
                    if incretin_effect_model == 'model1':
                        self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_oral_inc1.csv')
                    elif incretin_effect_model == 'model2':
                        self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_oral_inc2.csv')
                    elif incretin_effect_model == 'model3':
                        self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_oral_inc3.csv')
                    elif incretin_effect_model == 'model4':
                        self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_oral_inc4.csv')
                    else:
                        raise ValueError('Invalid value for incretin_effect_model')
                        
                else:
                    raise ValueError('Invalid value for default parameters')
            else:
                self.parameters = parameters
            self.incretin_effect_model = incretin_effect_model
    
                
        def __init__(self, Gb, Ib, parameters = 'normal', incretin_effect_model = 'model1'):
            self.init_model(Gb, Ib, parameters, incretin_effect_model)
        
        
        def simulation(self, t, d, plot = True):
            """
            t - time array in minutes
            d - carbohydrate intake array in mmol/min/kg
            """        
            Ts = t[1]-t[0] 
            idx_final = int(t[-1]/Ts)+1    
            x0 = np.array([self.Gb, 0, 0, 0, self.Ib, 0, 0])   
            x = np.zeros([idx_final, len(x0)])
            x[0,:] = x0   
            
            for i in range(1,idx_final):
                y = odeint(fcn_Bergman_extended, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                         args=(self.parameters,
                               0, d[i-1], 0, self.Gb, self.Ib,
                               self.incretin_effect_model, )
                         )
                x[i,:] = y[-1,:]
                
            if plot:
                plt.figure()
                plt.subplot(211)
                plt.plot(t, x[:,0])
                plt.xlabel('time [min]')
                plt.ylabel('glycemia [mmol/L]')
                plt.grid()
                plt.subplot(212)
                plt.plot(t, x[:,4])
                plt.xlabel('time [min]')
                plt.ylabel('plasma insulin [mU/L]')
                plt.grid()
                plt.tight_layout()
            return x


        def ogtt(self, glucose = 50, bodyweight = 70, plot = True):
            
            """
            glucose - amount of glucose intake for ogtt in grams
            bodyweight - subject bodyweight in kilograms
            """
            
            t = np.arange(0,180,1)
            d = np.zeros_like(t, dtype = float)
            d[0]=glucose * 1e3 / 180/ 1 / bodyweight
            x = self.simulation(t, d, plot)
            
            return x[:,0], x[:,4] # G, I


"""
Sorensen model of glucose metabolism 
glucose concentration are of a whole blood water

conversionas are as follows:
 blood glucose conc. = 84 % whole blood water glucose conc.
 plasma glucose conc. = 92.5 % whole blood water glucose conc.
 
"""
def fcn_Sorensen(x, t, p, Basal, r_IVG, r_IVI):
    # state signals
    G_BV, G_BI, G_H, G_G, G_L, G_K, G_PV, G_PI, MI_HGP, ff, MI_HGU, I_B, I_H, I_G, I_L, I_K, I_PV, I_PI, P, I, Q, Gamma = x 
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

        
class Sorensen_model:
    
    Gb = 0
    Ib = 0
    parameters = []
    x0 = []
    Basal = []
    def init_model(self, Gb, Ib, parameters = np.loadtxt(pathToParameters + 'par_Sorensen_normal.csv'), blood_measure = 'peripheral'):
        
        self.parameters = parameters
        
        alpha, beta, K, M1, M2, gamma = parameters   
    
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
            
        Q0 = 6.33
    
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
        
        if blood_measure == 'peripheral':
            GB_PV = Gb # use this if peripheral glucose is known
            GB_H  = rB_PGU/QG_P + GB_PV # use this if peripheral glucose is known
        elif blood_measure == 'arterial':
            GB_H = Gb # use this if arterial glucose is known
            GB_PV = GB_H - rB_PGU/QG_P # use this if arterial glucose is known
        else:
            raise ValueError("Wrong value for blood_measure")
        GB_PI = -TG_P/VG_PI*rB_PGU + GB_PV 
        GB_H  = rB_PGU/QG_P + GB_PV
        GB_K  = GB_H
        GB_G  = -r_GGU/QG_G + GB_H
        GB_L  = 1/QG_L * (QG_A * GB_H + QG_G * GB_G + rB_HGP - rB_HGU) 
        GB_BV = -r_BGU/QG_B + GB_H
        GB_BI = -r_BGU * TG_B/VG_BI + GB_BV
        
        if blood_measure == 'peripheral':
            IB_PV = Ib # use this if peripheral insulin is known
            IB_H = IB_PV/(1-0.15) # use this if peripheral insulin is known
        elif blood_measure == 'arterial':
            IB_H = Ib # use this if arterial insulin is known
            IB_PV= IB_H*(1-0.15) # use this if arterial insulin is known
        else:
            raise ValueError("Wrong value for blood_measure")
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
        self.x0 = [GB_BV, GB_BI, GB_H, GB_G, GB_L, GB_K, GB_PV, GB_PI, MIB_HGP, ffB, MIB_HGU, IB_B, IB_H, IB_G, IB_L, IB_K, IB_PV, IB_PI, PB, IB, QB, GammaB]
        self.Basal = [GB_BV, GB_BI, GB_H, GB_G, GB_L, GB_K, GB_PV, GB_PI, IB_B, IB_H, IB_G, IB_L, IB_K, IB_PV, IB_PI, GammaB, rB_PIR, SB]

    def __init__(self, Gb, Ib, parameters = np.loadtxt(pathToParameters + 'par_Sorensen_normal.csv'), blood_measure = 'peripheral'):
        self.init_model(Gb, Ib, parameters, blood_measure)   
        
    def simulation(self, t, r_IVG,r_IVI, plot = True):
        Ts = t[1]-t[0]
        idx_final = t.size
    
        x=np.zeros([idx_final,len(self.x0)])
        x[0,:] = self.x0

        for i in range(1,idx_final):
            y=odeint(fcn_Sorensen, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                     args=(self.parameters,
                           self.Basal,
                           r_IVG[i-1],r_IVI[i-1],), rtol=1e-5
                     )
            x[i,:] = y[-1,:]    
        
        if plot:
            plt.figure()
            plt.subplot(211)
            plt.plot(t, x[:,6]*0.8)
            plt.xlabel('time [min]')
            plt.ylabel('blood glucose [mmol/L]')
            plt.grid()
            plt.subplot(212)
            plt.plot(t, x[:,16])
            plt.xlabel('time [min]')
            plt.ylabel('plasma insulin [mU/L]')
            plt.grid()
            plt.tight_layout()
        
        return x
    
    def ivgtt(self, glucose_dose = 0.3, glucose_bolus_min = 2, insulin_dose = 20, insulin_bolus_min = 5, insulin_dosage_time = 20, plot = True ):
    
        """
        glucose_dose - glucose bolus in g/kg
        glucose_bolus_min - time during the glucose bolus is administred in minutes
        insulin_dose - insulin bolus in mU/kg
        insulin_bolus_min - time during the insulin bolus is administred in minutes
        insulin_dosage_time - time at which the insulin bolus is administred in minutes (from start of the ivgtt)
        """
        
        glucose_bolus = glucose_dose*1e3/180 # [mmol/kg]
        
        t = np.arange(0,180,1)
        r_IVG = np.zeros_like(t, dtype = float)
        r_IVG[0:int(glucose_bolus_min/1)] = glucose_bolus/1/glucose_bolus_min
        r_IVI = np.zeros_like(t, dtype = float)
        r_IVI[int(insulin_dosage_time/1):int(insulin_dosage_time/1)+int(insulin_bolus_min/1)] = insulin_dose/1/insulin_bolus_min
        
        x = self.simulation(t, r_IVG, r_IVI, plot)
        
        return x[:,6]*0.8, x[:,16] # G, I
        