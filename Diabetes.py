


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

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


def fcn_Bergman_extended(x, t, p, RaG_iv, d, RaI, Gb, Ib,
                         incretin_effect_model = 'model1'):
    
    G, X, D, Ra, I, v2, v3 = x    
    
    if incretin_effect_model == 'model1':
        
        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, Kg1, Kg2, T2, V_I, Kg1m, Kg2m = p  
        
        
        if d != 0 or Ra != 0:
            S = Kg1m*Kg1*(G-Gb) + Kg2m*Kg2/T2*(G-Gb)-v2
            v2_dot = -1/T2 * v2 + Kg2m*Kg2/(T2**2) * (G-Gb)
        else:
            S = Kg1*(G-Gb) + Kg2/T2*(G-Gb)-v2
            v2_dot = -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
        
        
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
        
    else:
        raise ValueError('Invalid value for incretin_effect_model')

    D_dot= -1/Td1 * D + Kd/Td1 * d
    Ra_dot= -1/Td2 * Ra + 1/Td2 * D

    G_dot, X_dot, I_dot = fcn_Bergman([G, X, I], 0, [T_G, Kx, V_G, T_X, T_I, V_I], (RaG_iv + Ra), RaI, S, Gb, Ib)
    
    return np.array([G_dot, X_dot, D_dot, Ra_dot, I_dot, v2_dot, v3_dot])


class minimal_model:
    
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
                    self.parameters = np.loadtxt(pathToParameters + 'par_normal_mean_inc1.csv')
                elif incretin_effect_model == 'model2':
                    self.parameters = np.loadtxt(pathToParameters + 'par_normal_mean_inc2.csv')
                elif incretin_effect_model == 'model3':
                    self.parameters = np.loadtxt(pathToParameters + 'par_normal_mean_inc3.csv')
                elif incretin_effect_model == 'model4':
                    self.parameters = np.loadtxt(pathToParameters + 'par_normal_mean_inc4.csv')
                else:
                    raise ValueError('Invalid value for incretin_effect_model')
                    
            elif parameters == 't2dm':
                if incretin_effect_model == 'model1':
                    self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_mean_inc1.csv')
                elif incretin_effect_model == 'model2':
                    self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_mean_inc2.csv')
                elif incretin_effect_model == 'model3':
                    self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_mean_inc3.csv')
                elif incretin_effect_model == 'model4':
                    self.parameters = np.loadtxt(pathToParameters + 'par_t2dm_mean_inc4.csv')
                else:
                    raise ValueError('Invalid value for incretin_effect_model')
                    
            else:
                raise ValueError('Invalid value for string parameters')
        else:
            self.parameters = parameters
        self.incretin_effect_model = incretin_effect_model

            
    def __init__(self, Gb, Ib, parameters = 'normal', incretin_effect_model = 'model1'):
        self.init_model(Gb, Ib, parameters, incretin_effect_model)
        
        
    def simulation(self, t, d, RaG_iv, RaI, plot = True):
        """
        t - time array in minutes
        d - carbohydrate intake array in mmol/min/kg
        RaG_iv - intravenous glucose infusion in mmol/min/kg
        RaI - intravenous insulin infusion in mU/min/kg
        """        
        Ts = t[1]-t[0] 
        idx_final = int(t[-1]/Ts)+1    
        x0 = np.array([self.Gb, 0, 0, 0, self.Ib, 0, 0])   
        x = np.zeros([idx_final, len(x0)])
        x[0,:] = x0   
        
        for i in range(1,idx_final):
            y = odeint(fcn_Bergman_extended, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                     args=(self.parameters,
                           RaG_iv[i-1], d[i-1], RaI[i-1], self.Gb, self.Ib,)
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
        return x


    def ogtt(self, glucose = 50, bodyweight = 70, plot = True):
        
        """
        glucose - amount of glucose intake for ogtt in grams
        bodyweight - subject bodyweight in kilograms
        """
        
        t = np.arange(0,180,1)
        RaG_iv = np.zeros_like(t, dtype = float)
        RaI = np.zeros_like(t, dtype = float)
        d = np.zeros_like(t, dtype = float)
        d[0]=glucose * 1e3 / 180/ 1 / bodyweight
        x = self.simulation(t, d, RaG_iv, RaI, plot)
        
        return x[:,0], x[:,4] # G, I
        
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
        d = np.zeros_like(t)
        RaG_iv = np.zeros_like(t, dtype = float)
        RaG_iv[0:int(glucose_bolus_min/1)] = glucose_bolus/1/glucose_bolus_min
        RaI = np.zeros_like(t, dtype = float)
        RaI[int(insulin_dosage_time/1):int(insulin_dosage_time/1)+int(insulin_bolus_min/1)] = insulin_dose/1/insulin_bolus_min
        
        x = self.simulation(t, d, RaG_iv, RaI, plot)
        
        return x[:,0], x[:,4] # G, I


Gb = 4.7
Ib = 6.7
model1 = minimal_model(Gb,Ib)

G, I = model1.ogtt()

mypars = model1.parameters

# Tg, Kx, V_G, Tx, Ti, Kg1, Kg2, T2, V_I = parGIh

T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, V_I, Kg1, Kg2, T2, Kg1m, Kg2m = model1.parameters

t = np.arange(0,120,1)
d = np.zeros_like(t)
d[0]=50*1e3/180/1/70

RaG_iv = np.zeros_like(t)
RaI = np.zeros_like(t)

g_dose = 0.3 # [g/kg]
glu_bolus = g_dose*1e3/180 # [mmol/kg]
glu_bolus_min = 2 
RaG_iv=np.zeros_like(t, dtype = float)
RaG_iv[0:int(glu_bolus_min/1)]=float(glu_bolus/1/glu_bolus_min)

i_dose = 20 # [mU/kg]
ins_bolus = i_dose # [mU/kg]
ins_bolus_min = 5 
RaI[int(20/1):int(20/1)+int(ins_bolus_min/1)]=ins_bolus/1/ins_bolus_min

x = model1.simulation(t, d, RaG_iv, RaI)

        