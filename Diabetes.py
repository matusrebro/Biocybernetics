


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



