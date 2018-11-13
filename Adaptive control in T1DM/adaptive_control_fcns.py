import numpy as np

from scipy.integrate import odeint

def fcn_DPmrac2(x,t,p,d,df,r,Gb,vb):
    x=np.reshape(x,(x.shape[0],1))
    D1, D2, S1, S2, I, x1, x2, x3, Q1, Q2, Gcgm = x[0:11]
    xm = x[11:13]
    vu = x[13:15]
    vy = x[15:17]
    omegaf = x[17:23]
    uf = x[23:24]
    xml = x[24:26]
    thetad = x[26]
    theta = x[27:]
    t_I, V_I, k_I, A_G, t_G, k_12, V_G, EGP_0, F_01, k_b1, k_b2, k_b3, k_a1, k_a2, k_a3, t_cgm = p


    # reference model parameters
    a0m=0.0003472
    a1m=0.05
    b0m=a0m
    
    Am=np.array([[0,1],[-a0m, -a1m]])
    bm=np.array([[0], [1]])
    cm=np.array([b0m, 0])
   
    # reference model output
    ym=np.dot(cm,xm)
    
    # Rate of appearance and glycemia signals
    Ra=D2/t_G
    G=Q1/V_G
    
    
    # deviation from basal state: feedback for controller
    y=Gcgm-Gb

    # auxiliary filter parameters
    lambda0=0.01
    lambda1=0.2    
    Al=np.array([[0,1],[-lambda0, -lambda1]])
    bl=np.array([[0], [1]])
    cl=np.array([[0, 1], [1, 0]])

    # filter outputs
    vuo=np.dot(cl,vu)   
    vyo=np.dot(cl,vy) 
    
    # omega signal for control law
    omega=np.zeros([6,1])
    omega[0:2,0]=vuo[:,0]
    omega[2:4,0]=vyo[:,0]
    omega[4:5,0]=y
    omega[5:,0]=r

    # parameter for SPR (Strictly Positive Real) reference model in adaptation law
    ro=0.01
    
    # a priori known sign of gain of our controlled system
    zn=-1
   
    cml=np.array([b0m*ro, b0m])
        
    # adaptation error
    ed=zn*np.dot(cml,xml)    
    e1=y-ym-ed
    
    # adaptation law parameters (Lyapunov part)
    Gamma=np.diag([0.1, 0.01, 0.1, 0.01, 50, 5e3])
    M0=100
    q0=1
    sigma0=1
    
    # sigma modification of adaptation law (Lyapunov part)
    sigmas=0
    if np.linalg.norm(theta)<=M0:
        sigmas=0
    elif (M0<np.linalg.norm(theta) and np.linalg.norm(theta)<=2*M0):
        sigmas=(np.linalg.norm(theta)/M0-1)**q0*sigma0
    elif np.linalg.norm(theta)>2*M0:
        sigmas=sigma0

    # adaptation law parameters (disturbance rejection part)
    gamma=5
    M0d=1000
    sigma0d=1e3

    # sigma modification of adaptation law (disturbance rejection part)    
    sigmad=0   
    if np.linalg.norm(thetad)<=M0d:
        sigmad=0
    elif (M0d<np.linalg.norm(thetad) and np.linalg.norm(thetad)<=2*M0d):
        sigmad=(np.linalg.norm(thetad)/M0d-1)**q0*sigma0d
    elif np.linalg.norm(thetad)>2*M0d:
        sigmad=sigma0d
      
    # control law
    u=np.dot(np.transpose(theta),omega)
    
    # disturbance rejection
    ud=thetad*df
    
    # saturation of control law output
    if u<=-vb:
        u=-vb
    
    if ud>0:
        ud=0
    
    # --- Hovorka model
    if G>=4.5:
        F_01c=F_01
    else:
        F_01c=F_01*G/4.5
        
    if G>=9:
        F_R=0.003*(G-9)*V_G
    else:
        F_R=0
 
    # diferential equations of Hovorka model
    D1_dot= A_G*d-D1/t_G
    D2_dot= D1/t_G-Ra
    S1_dot= (u+vb-ud)-S1/t_I
    S2_dot= S1/t_I-S2/t_I
    I_dot=  S2/(t_I*V_I)-k_I*I
    x1_dot= k_b1*I-k_a1*x1
    x2_dot= k_b2*I-k_a2*x2
    x3_dot= k_b3*I-k_a3*x3
    Q1_dot= -(F_01c+F_R)-x1*Q1+k_12*Q2+Ra+EGP_0*(1-x3)
    Q2_dot= x1*Q1-(k_12+x2)*Q2
    # CGM dynamics model
    Gcgm_dot=  G/t_cgm - Gcgm/t_cgm
    
    # reference model
    xm_dot=np.dot(Am,xm)+bm*r 
    # filters
    vu_dot=np.dot(Al,vu)+bl*u
    vy_dot=np.dot(Al,vy)+bl*y   
    # filtered omega
    omegaf_dot=-ro*omegaf+omega 
    # filtered control law output
    uf_dot=-ro*uf+u    
    # adaptation error augmentation signal
    xml_dot=np.dot(Am,xml)+bm*(uf-np.dot(np.transpose(theta),omegaf))  
    # adaptation law
    thetad_dot=-gamma*df*(y-ym)/(1+df**2)-sigmad*gamma*thetad
    theta_dot=-zn*e1*np.dot(Gamma,omegaf)/(1+np.dot(np.transpose(omegaf),omegaf))-sigmas*np.dot(Gamma,theta)
    
    # getting it all together for ODE solver
    x_dot=np.zeros(x.shape[0]) 
    x_dot[0:11] = D1_dot, D2_dot,S1_dot, S2_dot, I_dot, x1_dot, x2_dot, x3_dot, Q1_dot, Q2_dot, Gcgm_dot    
    x_dot[11:13] = np.squeeze(xm_dot)
    x_dot[13:15] = np.squeeze(vu_dot)
    x_dot[15:17] = np.squeeze(vy_dot) 
    x_dot[17:23] = np.squeeze(omegaf_dot)
    x_dot[23:24] = np.squeeze(uf_dot)
    x_dot[24:26] = np.squeeze(xml_dot)
    x_dot[26] = np.squeeze(thetad_dot)
    x_dot[27:] = np.squeeze(theta_dot)
    
    return x_dot

def sim_MRAC(t,p,dsigm,dsigc,r,Gb):
    t_I, V_I, k_I, A_G, t_G, k_12, V_G, EGP_0, F_01, k_b1, k_b2, k_b3, k_a1, k_a2, k_a3, t_cgm = p
    S_I1=k_b1/k_a1
    S_I2=k_b2/k_a2
    S_I3=k_b3/k_a3

    # basal insulin calculation for Hovorka model with basal glycemia Gb
    a=Gb*V_G*(S_I1*S_I2)+S_I2*S_I3*EGP_0
    b=-S_I2*(-F_01+EGP_0)+EGP_0*k_12*S_I3
    c=-k_12*(-F_01+EGP_0)   
    Ib1=(-b+np.sqrt(b**2-4*a*c))/2/a
    Ib2=(-b-np.sqrt(b**2-4*a*c))/2/a    
    Ibv=[Ib1, Ib2]
    Ib=[i for i in Ibv if i>0][0]
    vb=Ib*V_I*k_I
    
    # Hovorka model initial conditions
    D1b=0
    D2b=0
    S1b=vb*t_I
    S2b=S1b
    Ib=vb/(k_I*V_I)
    x1b=S_I1*Ib
    x2b=S_I2*Ib
    x3b=S_I3*Ib
    Q2b=(-F_01+EGP_0*(1-x3b))/x2b
    Q1b=Q2b*(k_12+x2b)/x1b
    Gb=Q1b/V_G
    Gcgmb=Gb

    Ts=t[1]-t[0] 
    idx_final=int(t[-1]/Ts)+1 

    thetad0=0
    x0=np.zeros(33)
    x0[0:11] = D1b, D2b, S1b, S2b, Ib, x1b, x2b, x3b, Q1b , Q2b, Gcgmb
    x0[26]=thetad0
    
    x=np.zeros([idx_final,33])
    x[0,:]=x0

    u=np.zeros([idx_final,1])
    ud=np.zeros([idx_final,1])

    cl=np.array([[0, 1], [1, 0]])
    
    dmax=np.max(dsigc)
    # --- adaptive control simulation
    for i in range(1,idx_final):
        df=0
        if dsigc[i-1,0]>0 and dsigc[i-1,0]<=1/3*dmax:
            df=1
        elif dsigc[i-1,0]>1/3*dmax and dsigc[i-1,0]<=2/3*dmax:
            df=2
        elif dsigc[i-1,0]>2/3*dmax:
            df=3
        y=odeint(fcn_DPmrac2,x[i-1,:],np.linspace((i-1)*Ts,i*Ts,11),
                 args=(p,
                       dsigm[i-1,0],
                       df,
                       r[i-1,0],
                       Gb,
                       vb,)
                )
        x[i,:] = y[-1,:]
        vu = np.transpose(x[i,13:15])
        vy = np.transpose(x[i,15:17])
        vuo = np.dot(cl,vu) 
        vyo = np.dot(cl,vy) 
        omega=np.zeros([6,1])
        omega[0:2,0]=vuo
        omega[2:4,0]=vyo
        omega[4:5,0]=x[i,10]
        omega[5:,0]=r[i,0]
        theta=x[i,27:]
        thetad=x[i,26]
        u[i,0]=np.dot(theta,omega)
        if u[i,0]<=-vb:
            u[i,0]=-vb
        
        ud[i,0]=thetad*df
        if ud[i,0]>0:
            ud[i,0]=0
    
    # x - state vector
    # u - basal insulin administration [mU/min]
    # ud - disturbance rejection (negative value of bolus administration) [mU/min]
    # vb - basal insulin administration corresponding to given basal glycemia
    return x, u, ud, vb