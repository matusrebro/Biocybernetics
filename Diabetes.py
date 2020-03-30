


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
        

