


import numpy as np
from scipy.integrate import odeint


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
        
        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, V_I, Kg1, Kg2, T2, Kg1m, Kg2m = p  
        
        
        S = Kg1m*Kg1*(G-Gb) + Kg2m*Kg2/T2*(G-Gb)-v2
        
        v2_dot = -1/T2 * v2 + Kg2m*Kg2/(T2**2) * (G-Gb)
        v3_dot = 0
                
    elif incretin_effect_model == 'model2':

        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, V_I, Kg1, Kg2, T2, Kg3a, Kg3b = p
        
        S = Kg1*(G-Gb) + Kg2/T2*(G-Gb)-v2 + Kg3a*D + Kg3b*Ra
        
        v2_dot = -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
        v3_dot = 0

     
    elif incretin_effect_model == 'model3':
        
        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, V_I, Kg1, Kg2, T2, T3, Kg3a, Kg3b = p  
        
        S = Kg1*(G-Gb) + Kg2/T2*(G-Gb) - v2 + v3
        
        v2_dot= -1/T2 * v2 + Kg2/(T2**2) * (G-Gb)
        v3_dot= -1/T3 * v3 + Kg3a*D/T3 + Kg3b*Ra/T3
        
    elif incretin_effect_model == 'model4' :
        
        T_G, Kx, V_G, T_X, Kd, Td1, Td2, T_I, V_I, Kg1, Kg2, T2, Kg1m, Kg2m, T3, Kg3a, Kg3b = p 
        
        S = (1+Kg1m*v3)*Kg1*(G-Gb) + (1+Kg2m*v3)*Kg2/T2*(G-Gb)-v2
        
        v2_dot= -1/T2 * v2 + (1+Kg2m*v3)*Kg2/(T2**2) * (G-Gb)
        v3_dot= -1/T3 * v3 + Kg3a*D/T3 + Kg3b*Ra/T3
        
    else:
        raise ValueError('Invalid value for incretin_effect_model')

    D_dot= -1/Td1 * D + Kd/Td1 * d
    Ra_dot= -1/Td2 * Ra + 1/Td2 * D

    G_dot, X_dot, I_dot = fcn_Bergman(x, 0, [T_G, Kx, V_G, T_X, T_I, V_I], RaG_iv, RaI, S, Gb, Ib)
    
    return np.array([G_dot, X_dot, D_dot, Ra_dot, I_dot, v2_dot, v3_dot])


class minimal_model:
    
    Gb = 0
    Ib = 0
    parameters = []
    incretin_effect_model = ''
    
    def init_model(self, Gb, Ib, parameters, incretin_effect_model = 'model1'):
        
        self.Gb = Gb
        self.Ib = Ib
        self.parameters = parameters
        self.incretin_effect_model = incretin_effect_model

            
    def __init__(self, Gb, Ib, parameters, incretin_effect_model = 'model1'):
        self.set_parameters(Gb, Ib, parameters, incretin_effect_model)
        
        
    def simulation(self, t, RaG_iv, d, RaI):
        
        Ts = t[1]-t[0] 
        idx_final = int(t[-1]/Ts)+1    
        x0 = np.array([self.Gb, 0, 0, 0, self.Ib, 0, 0])   
        x = np.zeros([idx_final, len(x0)])
        x[0,:] = x0   
        
        for i in range(1,idx_final):
            y=odeint(fcn_Bergman_extended, x[i-1,:], np.linspace((i-1)*Ts,i*Ts),
                     args=(self.parameters,
                           d[i-1], self.Gb, self.Ib,)
                     )
            x[i,:] = y[-1,:]
        return x
        