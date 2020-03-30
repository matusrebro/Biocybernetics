# functions for short-term glycemia prediction on the basis of CGM data

import numpy as np
import matplotlib.pyplot as plt

# Clarke EGA (Error Grid Analysis) plot
# returns percentages of points in A,B,C,D,E areas
def clarkeEGA(G,GpredN,k,N,filename):
    # G - array of glycemia measurements (CGM)
    # GpredN - array of glycemia predictions
    # k - number of samples we ommit from comparison due to time it takes to paramaters to settle
    # N - prediction horizon (number of steps we predict ahead)
    # file name to save the EGA graph to
    if N>1:       
        yEGA=G[N-1+k:]
        ypEGA=GpredN[k:-N+1]
    else:
        yEGA=G[N-1+k:]
        ypEGA=GpredN[k:]
        
    total=np.zeros([5,1])
    
    for i in range(len(yEGA)):
        if (ypEGA[i]<=70/18 and yEGA[i]<=70/18) or (ypEGA[i]<=1.2*yEGA[i] and ypEGA[i]>=0.8*yEGA[i]):
            total[0,0]=total[0,0]+1
        elif (yEGA[i]>=180/18 and ypEGA[i]<=70/18) or (yEGA[i]<=70/18 and ypEGA[i]>=180/18):
            total[4,0]=total[4,0]+1
        elif ((yEGA[i]>=70/18 and yEGA[i]<=290/18) and (ypEGA[i]>=yEGA[i]+110/18)) or ((yEGA[i]>=130/18 and yEGA[i]<=180/18) and (ypEGA[i]<=7/5*yEGA[i]-182/18)):
            total[2,0]=total[2,0]+1
        elif (yEGA[i]>=240/18 and (ypEGA[i]>=70/18 and ypEGA[i]<=180/18)) or (yEGA[i]<=175/3/18 and ypEGA[i]<=180/18 and ypEGA[i]>=70/18) or ((yEGA[i]>=175/3/18 and yEGA[i]<=70/18) and ypEGA[i]>=6/5*yEGA[i]):
            total[3,0]=total[3,0]+1
        else:
            total[1,0]=total[1,0]+1
    
    percentage=total/len(yEGA)*100
    
    plt.figure()
    plt.axis([0, 400/18, 0, 400/18])
    plt.plot([0,400/18],[0,400/18],'k:')
    plt.plot([0, 175/3/18],[70/18, 70/18],'k-')
    plt.plot([175/3/18, 400/1.2/18],[70/18, 400/18],'k-')
    plt.plot([70/18, 70/18],[84/18, 400/18],'k-')
    plt.plot([0, 70/18],[180/18, 180/18],'k-')
    plt.plot([70/18, 290/18],[180/18, 400/18],'k-') 
    plt.plot([70/18, 70/18],[0, 56/18],'k-')  
    plt.plot([70/18, 400/18],[56/18, 320/18],'k-')
    plt.plot([180/18, 180/18],[0, 70/18],'k-')
    plt.plot([180/18, 400/18],[70/18, 70/18],'k-')
    plt.plot([240/18, 240/18],[70/18, 180/18],'k-')
    plt.plot([240/18, 400/18],[180/18, 180/18],'k-')
    plt.plot([130/18, 180/18],[0, 70/18],'k-')   
    plt.text(30/18,15/18,'A',fontsize=15)
    plt.text(30/18,150/18,'D',fontsize=15)
    plt.text(30/18,360/18,'E',fontsize=15)
    plt.text(150/18,360/18,'C',fontsize=15)
    plt.text(160/18,20/18,'C',fontsize=15)
    plt.text(380/18,20/18,'E',fontsize=15)
    plt.text(380/18,120/18,'D',fontsize=15)
    plt.text(380/18,260/18,'B',fontsize=15)
    plt.text(280/18,360/18,'B',fontsize=15)
    plt.plot(yEGA,ypEGA,'kx')   
    plt.xlabel(u'measured glycemia [mmol/L]')
    plt.ylabel(u'predicted glycemia [mmol/L]')        
    plt.savefig(filename+'.pdf')
    plt.close('all')
    print(np.transpose(percentage))
    return percentage

# similar to EGA but for comparison of glycemia rates
def clarkeEGArate(G,GpredN,k,N,Ts,filename):
    # G - array of glycemia measurements (CGM)
    # GpredN - array of glycemia predictions
    # k - number of samples we ommit from comparison due to time it takes to paramaters to settle
    # Ts - sample time
    # N - prediction horizon (number of steps we predict ahead)
    # file name to save the EGA graph to
    dG=np.gradient(np.squeeze(G),Ts)
    dGpredN=np.gradient(np.squeeze(GpredN),Ts)    
    
    if N>1:       
        yEGA=dG[N-1+k:]
        ypEGA=dGpredN[k:-N+1]
    else:
        yEGA=dG[N-1+k:]
        ypEGA=dGpredN[k:]
        
    total=np.zeros([5,1])
    
    for i in range(len(yEGA)):
        if (np.abs(ypEGA[i])>=2/18 and np.abs(yEGA[i])>=2/18) or (np.abs(ypEGA[i])<=2*np.abs(yEGA[i]) and np.abs(yEGA[i])<=2/18 and np.abs(yEGA[i])>=1/18 and np.abs(ypEGA[i]>=2/18)) or (np.abs(yEGA[i])<=2*np.abs(ypEGA[i]) and np.abs(ypEGA[i])<=2/18 and np.abs(ypEGA[i])>=1/18 and np.abs(yEGA[i]>=2/18)) or (ypEGA[i]>=yEGA[i]-1/18 and yEGA[i]>=ypEGA[i]-1/18):
            total[0,0]=total[0,0]+1
        elif (yEGA[i]>=1/18 and ypEGA[i]<=-1/18) or (yEGA[i]<=-1/18 and ypEGA[i]>=1/18):
            total[4,0]=total[4,0]+1
        elif (np.abs(yEGA[i])<=1/18 and np.abs(ypEGA[i])>=np.abs(yEGA[i])+2/18):
            total[2,0]=total[2,0]+1
#        elif (np.abs(ypEGA[i]<=1/18) and np.abs(ypEGA[i])>=np.abs(yEGA[i])+2/18):
        elif (np.abs(ypEGA[i]<=1/18) and (ypEGA[i]>=yEGA[i]+2/18 or ypEGA[i]<=yEGA[i]-2/18)):
            total[3,0]=total[3,0]+1
        else:
            total[1,0]=total[1,0]+1
    
    percentage=total/len(yEGA)*100
    
    plt.axis([-4/18, 4/18, -4/18, 4/18])
    plt.plot([-4/18, 4/18],[-4/18, 4/18],'k:')
    plt.plot([-2/18, -1/18],[-4/18, -2/18],'k-')
    plt.plot([-1/18, 2/18],[-2/18, 1/18],'k-')
    plt.plot([2/18, 4/18],[1/18, 2/18],'k-')
    plt.plot([-4/18, -2/18],[-2/18, -1/18],'k-')
    plt.plot([-2/18, 1/18],[-1/18, 2/18],'k-') 
    plt.plot([1/18, 2/18],[2/18, 4/18],'k-')        
    plt.plot([-1/18, -1/18],[-4/18, -3/18],'k-')
    plt.plot([-1/18, 3/18],[-3/18, 1/18],'k-')
    plt.plot([3/18, 4/18],[1/18, 1/18],'k-')
    plt.plot([-4/18, -3/18],[-1/18, -1/18],'k-')
    plt.plot([-3/18, 1/18],[-1/18, 3/18],'k-')
    plt.plot([1/18, 1/18],[3/18, 4/18],'k-')
    plt.plot([-4/18, -1/18],[1/18, 1/18],'k-')
    plt.plot([-1/18, -1/18],[1/18, 4/18],'k-')
    plt.plot([1/18, 1/18],[-4/18, -1/18],'k-')
    plt.plot([1/18, 4/18],[-1/18, -1/18],'k-')
    plt.text(-3/18,-3/18,'A',fontsize=15)
    plt.text(-1.5/18,-3.8/18,'B',fontsize=15)
    plt.text(0/18,-3.8/18,'C',fontsize=15)
    plt.text(3.5/18,-3.8/18,'E',fontsize=15)
    plt.text(-3.5/18,-1.5/18,'B',fontsize=15)
    plt.text(3.5/18,-0.5/18,'D',fontsize=15)
    plt.text(-3.5/18,0.5/18,'D',fontsize=15)
    plt.text(-3.5/18,3.5/18,'E',fontsize=15)
    plt.text(0/18,3.5/18,'C',fontsize=15)
    plt.plot(yEGA,ypEGA,'kx')   
    plt.xlabel('measured glycemia rate [mmol/L/min]')
    plt.ylabel('predicted glycemia rate [mmol/L/min]')
    plt.savefig(filename+'.pdf')
    plt.close('all')
    print(np.transpose(percentage))
    return percentage
    
# prediction N-steps ahead using ARX (2 inputs) and recursive least squares algorithm for parameter identification
def Gpredarx2(G,Gpb,dsig,vsig,na,nb1,nb2,N,fz,mao):
    # G - array of CGM data
    # Gpb - glycemia operating point (estimation of basal glycemia)
    # dsig - 1st input - rate of carbohydrate intake
    # vsig - 2nd input - rate of insulin administration
    # na - order of polynomial of autoregressive part of model
    # nb1 - order of polynomial of 1st input part of model
    # nb2 - order of polynomial of 2nd input part of model
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    # mao - order of MA filter for CGM data (if mao=0 - means no filter is used)
    idx_final=G.shape[0]
    y1=np.zeros([idx_final,1])
    Gf=np.zeros([idx_final,1])
    Gf[0,0]=G[0]
    if mao>0:
        ypf=np.zeros([mao,1])
        ypf[0:mao,0]=G[0:mao,0]

    yp=np.zeros([na,1])
    up1=np.zeros([nb1,1])
    up2=np.zeros([nb2,1])
    theta=np.zeros([idx_final,na+nb1+nb2])
    P=np.eye(na+nb1+nb2)*1e6
    
    ypredN=np.zeros([idx_final,1])
    GpredN=np.zeros([idx_final,1])

    for i in range(0,idx_final):
        y1[i,0]=G[i]-Gpb
        if mao>0:
            if i>0:
                ypf=np.roll(ypf,1)
                ypf[0,0]=G[i]         
                Gf[i]=1/(mao+1)*(G[i]+np.sum(ypf))
            y1[i,0]=Gf[i]-Gpb

        
        for c in range(1,na+1):
            if i-c>=0:
                yp[c-1,0]=y1[i-c]
            else:
                yp[c-1,0]=0
                
        for c in range(1,nb1+1):
            if i-c>=0:
                up1[c-1,0]=vsig[i-c]-vsig[0]
            else:
                up1[c-1,0]=0 
                
        for c in range(1,nb2+1):
            if i-c>=0:
                up2[c-1,0]=dsig[i-c]
            else:
                up2[c-1,0]=0        
                
        h=np.vstack((-yp,up1))
        h=np.vstack((h,up2))
                
        e=y1[i,0]-np.dot(h[:,0],theta[i-1,:])
        Y=np.dot(P,h)/(1+np.dot(np.dot(np.transpose(h),P),h))
        P=(P-np.dot(np.dot(Y,np.transpose(h)),P))*1/fz
        theta[i,:]=theta[i-1,:]+np.transpose(np.dot(Y,e))
        thetak=np.transpose(theta[i,:])
    
        ypred=np.zeros([N,1])
        yp=np.roll(yp,1)
        yp[0,0]=y1[i,0]
        for k in range(N):
            
            if k>0:
                yp=np.roll(yp,1)
                yp[0,0]=ypred[k-1] 
                up1=np.roll(up1,1)
                up1[0,0]=0            
                up2=np.roll(up2,1)
                up2[0,0]=0      
                            
            h=np.vstack((-yp,up1))
            h=np.vstack((h,up2)) 
        
            ypred[k,0]=np.dot(h[:,0],thetak)
            
        ypredN[i,0]=ypred[-1,0]
        GpredN[i,0]=ypredN[i,0]+Gpb

    return GpredN,theta,thetak,Gf
    

# prediction N-steps ahead using ARX (1 input) and recursive least squares algorithm for parameter identification
def Gpredarx1(G,Gpb,vsig,na,nb1,N,fz,mao):
    # G - array of CGM data
    # Gpb - glycemia operating point (estimation of basal glycemia)
    # vsig - input - rate of insulin administration
    # na - order of polynomial of autoregressive part of model
    # nb1 - order of polynomial of input part of model
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    # mao - order of MA filter for CGM data (if mao=0 - means no filter is used)
    idx_final=G.shape[0]
    y1=np.zeros([idx_final,1])
    Gf=np.zeros([idx_final,1])
    Gf[0,0]=G[0]
    if mao>0:
        ypf=np.zeros([mao,1])
        ypf[0:mao,0]=G[0:mao,0]
            
    yp=np.zeros([na,1])
    up1=np.zeros([nb1,1])
    theta=np.zeros([idx_final,na+nb1])
    P=np.eye(na+nb1)*1e6        

    ypredN=np.zeros([idx_final,1])
    GpredN=np.zeros([idx_final,1])        
    
    
    for i in range(0,idx_final):
        y1[i,0]=G[i]-Gpb

        if mao>0:
            if i>0:
                ypf=np.roll(ypf,1)
                ypf[0,0]=G[i]
            
                Gf[i]=1/(mao+1)*(G[i]+np.sum(ypf))
            y1[i,0]=Gf[i]-Gpb
        
        for c in range(1,na+1):
            if i-c>=0:
                yp[c-1,0]=y1[i-c]
            else:
                yp[c-1,0]=0
                
        for c in range(1,nb1+1):
            if i-c>=0:
                up1[c-1,0]=vsig[i-c]-vsig[0]
            else:
                up1[c-1,0]=0 
         
        h=np.vstack((-yp,up1))
               
        e=y1[i,0]-np.dot(h[:,0],theta[i-1,:])
        Y=np.dot(P,h)/(1+np.dot(np.dot(np.transpose(h),P),h))
        P=(P-np.dot(np.dot(Y,np.transpose(h)),P))*1/fz
        theta[i,:]=theta[i-1,:]+np.transpose(np.dot(Y,e))
        thetak=np.transpose(theta[i,:])
    
        ypred=np.zeros([N,1])
        yp=np.roll(yp,1)
        yp[0,0]=y1[i,0]
        for k in range(N):
            
            if k>0:
                yp=np.roll(yp,1)
                yp[0,0]=ypred[k-1] 
                up1=np.roll(up1,1)
                up1[0,0]=0       
                            
            h=np.vstack((-yp,up1))
        
            ypred[k,0]=np.dot(h[:,0],thetak)
            
        ypredN[i,0]=ypred[-1,0]
        GpredN[i,0]=ypredN[i,0]+Gpb
 
    return GpredN,theta,thetak,Gf
    
# prediction N-steps ahead using ARMA (no input) and recursive least squares algorithm for parameter identification    
def Gpredarma(G,Gpb,na,nc,N,fz,mao):
    # G - array of CGM data
    # Gpb - glycemia operating point (estimation of basal glycemia)
    # vsig - input - rate of insulin administration
    # na - order of polynomial of autoregressive part of model
    # nc - order of polynomial of moving average part of model
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    # mao - order of MA filter for CGM data (if mao=0 - means no filter is used)
    idx_final=G.shape[0]
    y1=np.zeros([idx_final,1])
    e1=np.zeros([idx_final,1])
    Gf=np.zeros([idx_final,1])
    Gf[0,0]=G[0]
    if mao>0:
        ypf=np.zeros([mao,1])
        ypf[0:mao,0]=G[0:mao,0]
        
    yp=np.zeros([na,1])
    ep=np.zeros([nc,1])
    theta=np.zeros([idx_final,na+nc])
    P=np.eye(na+nc)*1e6
    
    ypredN=np.zeros([idx_final,1])
    GpredN=np.zeros([idx_final,1])    
    

    for i in range(0,idx_final):  
        y1[i,0]=G[i]-Gpb

        if mao>0:
            if i>0:
                ypf=np.roll(ypf,1)
                ypf[0,0]=G[i]
            
                Gf[i]=1/(mao+1)*(G[i]+np.sum(ypf))
            y1[i,0]=Gf[i]-Gpb
        
        for c in range(1,na+1):
            if i-c>=0:
                yp[c-1,0]=y1[i-c]
            else:
                yp[c-1,0]=0
                
        for c in range(1,nc+1):
            if i-c>=0:
                ep[c-1,0]=e1[i-c]
            else:
                ep[c-1,0]=0 
                   
        h=np.vstack((-yp,ep))
        
        e=y1[i,0]-np.dot(h[:,0],theta[i-1,:])
        e1[i,0]=e
        Y=np.dot(P,h)/(1+np.dot(np.dot(np.transpose(h),P),h))
        P=(P-np.dot(np.dot(Y,np.transpose(h)),P))*1/fz
        theta[i,:]=theta[i-1,:]+np.transpose(np.dot(Y,e))
        thetak=np.transpose(theta[i,:])
    
        yp=np.roll(yp,1)
        yp[0,0]=y1[i,0]


        ypred=np.zeros([N,1])
        for k in range(N):
            
            if k>0:
                yp=np.roll(yp,1)
                yp[0,0]=ypred[k-1] 
                ep=np.roll(ep,1)
                ep[0,0]=0                           
            h=np.vstack((-yp,ep))
        
            ypred[k,0]=np.dot(h[:,0],thetak)
            
        ypredN[i,0]=ypred[-1,0]
        GpredN[i,0]=ypredN[i,0]+Gpb

    return GpredN,theta,thetak,Gf

# prediction N-steps ahead using ARMAX (2 inputs) and recursive least squares algorithm for parameter identification    
def Gpredarmax2(G,Gpb,dsig,vsig,na,nb1,nb2,nc,N,fz,mao):
    # G - array of CGM data
    # Gpb - glycemia operating point (estimation of basal glycemia)
    # vsig - input - rate of insulin administration
    # na - order of polynomial of autoregressive part of model
    # dsig - 1st input - rate of carbohydrate intake
    # vsig - 2nd input - rate of insulin administration
    # na - order of polynomial of autoregressive part of model
    # nb1 - order of polynomial of 1st input part of model
    # nb2 - order of polynomial of 2nd input part of model
    # nc - order of polynomial of moving average part of model
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    # mao - order of MA filter for CGM data (if mao=0 - means no filter is used)

    idx_final=G.shape[0]
    y1=np.zeros([idx_final,1])
    e1=np.zeros([idx_final,1])

    Gf=np.zeros([idx_final,1])
    Gf[0,0]=G[0]
    if mao>0:
        ypf=np.zeros([mao,1])
        ypf[0:mao,0]=G[0:mao,0]
        
    yp=np.zeros([na,1])
    up1=np.zeros([nb1,1])
    up2=np.zeros([nb2,1])
    ep=np.zeros([nc,1])
    
    theta=np.zeros([idx_final,na+nb1+nb2+nc])
    P=np.eye(na+nb1+nb2+nc)*1e6
    
    ypredN=np.zeros([idx_final,1])
    GpredN=np.zeros([idx_final,1])    
        
    for i in range(0,idx_final):
    
        y1[i,0]=G[i]-Gpb

        if mao>0:
            if i>0:
                ypf=np.roll(ypf,1)
                ypf[0,0]=G[i]

                Gf[i]=1/(mao+1)*(G[i]+np.sum(ypf))
            y1[i,0]=Gf[i]-Gpb

        
        for c in range(1,na+1):
            if i-c>=0:
                yp[c-1,0]=y1[i-c]
            else:
                yp[c-1,0]=0
    
    
        for c in range(1,nb1+1):
            if i-c>=0:
                up1[c-1,0]=vsig[i-c]-vsig[0]
#                up1[c-1,0]=vsig[i-c]
            else:
                up1[c-1,0]=0 
                
        for c in range(1,nb2+1):
            if i-c>=0:
                up2[c-1,0]=dsig[i-c]
            else:
                up2[c-1,0]=0   
    
              
        for c in range(1,nc+1):
            if i-c>=0:
                ep[c-1,0]=e1[i-c]
            else:
                ep[c-1,0]=0 
                
     
        h=np.vstack((-yp,up1))      
        h=np.vstack((h,up2))
        h=np.vstack((h,ep))
                
        e=y1[i,0]-np.dot(h[:,0],theta[i-1,:])
        e1[i,0]=e
        Y=np.dot(P,h)/(1+np.dot(np.dot(np.transpose(h),P),h))
        P=(P-np.dot(np.dot(Y,np.transpose(h)),P))*1/fz
        theta[i,:]=theta[i-1,:]+np.transpose(np.dot(Y,e))
        thetak=np.transpose(theta[i,:])   


        ypred=np.zeros([N,1])
        yp=np.roll(yp,1)
        yp[0,0]=y1[i,0]
        for k in range(N):
            
            if k>0:
                yp=np.roll(yp,1)
                yp[0,0]=ypred[k-1] 
                up1=np.roll(up1,1)
                up1[0,0]=0            
                up2=np.roll(up2,1)
                up2[0,0]=0             
                ep=np.roll(ep,1)
                ep[0,0]=0      
                          
            h=np.vstack((-yp,up1))      
            h=np.vstack((h,up2))
            h=np.vstack((h,ep))
        
            ypred[k,0]=np.dot(h[:,0],thetak)
            
        ypredN[i,0]=ypred[-1,0]
        GpredN[i,0]=ypredN[i,0]+Gpb
    
    return GpredN,theta,thetak,Gf
    
# prediction N-steps ahead using ARMAX (1 input) and recursive least squares algorithm for parameter identification    
def Gpredarmax1(G,Gpb,vsig,na,nb,nc,N,fz,mao):
    # G - array of CGM data
    # Gpb - glycemia operating point (estimation of basal glycemia)
    # vsig - input - rate of insulin administration
    # na - order of polynomial of autoregressive part of model
    # dsig - 1st input - rate of carbohydrate intake
    # vsig - 2nd input - rate of insulin administration
    # na - order of polynomial of autoregressive part of model
    # nb1 - order of polynomial of input part of model
    # nc - order of polynomial of moving average part of model
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    # mao - order of MA filter for CGM data (if mao=0 - means no filter is used)

    idx_final=G.shape[0]
    y1=np.zeros([idx_final,1])
    e1=np.zeros([idx_final,1])

    Gf=np.zeros([idx_final,1])
    Gf[0,0]=G[0]
    if mao>0:
        ypf=np.zeros([mao,1])
        ypf[0:mao,0]=G[0:mao,0]
        
    yp=np.zeros([na,1])
    up=np.zeros([nb,1])
    ep=np.zeros([nc,1])
    
    theta=np.zeros([idx_final,na+nb+nc])
    P=np.eye(na+nb+nc)*1e6
    
    ypredN=np.zeros([idx_final,1])
    GpredN=np.zeros([idx_final,1])
    

    for i in range(0,idx_final):
    
        y1[i,0]=G[i]-Gpb

        if mao>0:
            if i>0:
                ypf=np.roll(ypf,1)
                ypf[0,0]=G[i]

                Gf[i]=1/(mao+1)*(G[i]+np.sum(ypf))
            y1[i,0]=Gf[i]-Gpb

        
        for c in range(1,na+1):
            if i-c>=0:
                yp[c-1,0]=y1[i-c]
            else:
                yp[c-1,0]=0
    
    
        for c in range(1,nb+1):
            if i-c>=0:
                up[c-1,0]=vsig[i-c]-vsig[0]
            else:
                up[c-1,0]=0 
    
    
        for c in range(1,nc+1):
            if i-c>=0:
                ep[c-1,0]=e1[i-c]
            else:
                ep[c-1,0]=0 
                
        h=np.vstack((-yp,up))        
        h=np.vstack((h,ep))
        
        e=y1[i,0]-np.dot(h[:,0],theta[i-1,:])
        e1[i,0]=e
        Y=np.dot(P,h)/(1+np.dot(np.dot(np.transpose(h),P),h))
        P=(P-np.dot(np.dot(Y,np.transpose(h)),P))*1/fz
        theta[i,:]=theta[i-1,:]+np.transpose(np.dot(Y,e))
        thetak=np.transpose(theta[i,:])   


        ypred=np.zeros([N,1])
        yp=np.roll(yp,1)
        yp[0,0]=y1[i,0]
        for k in range(N):
            
            if k>0:
                yp=np.roll(yp,1)
                yp[0,0]=ypred[k-1] 
                up=np.roll(up,1)
                up[0,0]=0             
                ep=np.roll(ep,1)
                ep[0,0]=0      
                          
            h=np.vstack((-yp,up))        
            h=np.vstack((h,ep))
        
            ypred[k,0]=np.dot(h[:,0],thetak)
            
        ypredN[i,0]=ypred[-1,0]
        GpredN[i,0]=ypredN[i,0]+Gpb

    return GpredN,theta,thetak,Gf

# prediction N-steps ahead using Box-Jenkins (2 inputs) and recursive least squares algorithm for parameter identification    
# y = B1/A1 * vsig + B2/A2 * dsig + C/D * e
def GpredBJ2(G,Gpb,dsig,vsig,na1,na2,nb1,nb2,nc,nd,N,fz,mao):
    # G - array of CGM data
    # Gpb - glycemia operating point (estimation of basal glycemia)
    # vsig - input - rate of insulin administration
    # na1 - order of polynomial A1
    # na2 - order of polynomial A2
    # nb1 - order of polynomial B1
    # nb2 - order of polynomial B2
    # nc - order of polynomial C
    # nd - order of polynomial D
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    # mao - order of MA filter for CGM data (if mao=0 - means no filter is used)
    idx_final=G.shape[0]
    y1=np.zeros([idx_final,1])
    e1=np.zeros([idx_final,1])

    Gf=np.zeros([idx_final,1])
    Gf[0,0]=G[0]
    if mao>0:
        ypf=np.zeros([mao,1])
        ypf[0:mao,0]=G[0:mao,0]

    xp1=np.zeros([na1,1])
    xp2=np.zeros([na2,1])
    up1=np.zeros([nb1,1])
    up2=np.zeros([nb2,1])
    wp=np.zeros([nc,1])
    vp=np.zeros([nd,1])
    
    theta1=np.zeros([idx_final,na1+nb1])
    theta2=np.zeros([idx_final,na2+nb2])
    theta3=np.zeros([idx_final,nc+nd])
    theta=np.zeros([idx_final,na1+na2+nb1+nb2+nc+nd])
    P=np.eye(na1+na2+nb1+nb2+nc+nd)*1e6
    
    h1=np.vstack((-xp1,up1))  
    h2=np.vstack((-xp2,up2))    
    h3=np.vstack((-wp,vp))   
    h=np.vstack((h1,h2))
    h=np.vstack((h,h3))
    
    ypredN=np.zeros([idx_final,1])
    GpredN=np.zeros([idx_final,1])

    for i in range(0,idx_final):    
        y1[i,0]=G[i]-Gpb

        if mao>0:
            if i>0:
                ypf=np.roll(ypf,1)
                ypf[0,0]=G[i]

                Gf[i]=1/(mao+1)*(G[i]+np.sum(ypf))
            y1[i,0]=Gf[i]-Gpb
        
    #    for c in range(1,na+1):
    #        if i-c>=0:
    #            yp[c-1,0]=y1[i-c]
    #        else:
    #            yp[c-1,0]=0
        
    #    for c in range(1,na1+1):
    #        if i-c>=0:
    #            xp1[c-1,0]=np.dot(h1[:,0],theta1[i-1,:])
    #        else:
    #            xp1[c-1,0]=0
           
        xp1=np.roll(xp1,1)
        xp1[0,0]=np.dot(h1[:,0],theta1[i-1,:])
        
    #    for c in range(1,na2+1):
    #        if i-c>=0:
    #            xp2[c-1,0]=np.dot(h2[:,0],theta2[i-1,:])
    #        else:
    #            xp2[c-1,0]=0
    
        xp2=np.roll(xp2,1)
        xp2[0,0]=np.dot(h2[:,0],theta2[i-1,:])
    
        for c in range(1,nb1+1):
            if i-c>=0:
                up1[c-1,0]=vsig[i-c]-vsig[0]
            else:
                up1[c-1,0]=0 
                
        
        for c in range(1,nb2+1):
            if i-c>=0:
                up2[c-1,0]=dsig[i-c]
            else:
                up2[c-1,0]=0   
    
              
    #    for c in range(1,nc+1):
    #        if i-c>=0:
    #            ep[c-1,0]=e1[i-c]
    #        else:
    #            ep[c-1,0]=0 
    
        wp=np.roll(wp,1)
        wp[0,0]=y1[i,0]-(np.dot(h1[:,0],theta1[i-1,:])+np.dot(h2[:,0],theta2[i-1,:]))
        
        vp=np.roll(vp,1)
        vp[0,0]=y1[i,0]-np.dot(h[:,0],theta[i-1,:])         
        
    #    h1=np.vstack((-xp1,up1))  
    #    h2=np.vstack((-xp2,up2))    
    #    h3=np.vstack((-wp,vp))   
    #    h=np.vstack((h1,h2))
    #    h=np.vstack((h,h3))
        
    #    H[i,:]=np.transpose(h)    
        
        e=y1[i,0]-np.dot(h[:,0],theta[i-1,:])
        e1[i,0]=e
        Y=np.dot(P,h)/(1+np.dot(np.dot(np.transpose(h),P),h))
        P=(P-np.dot(np.dot(Y,np.transpose(h)),P))*1/fz
        theta[i,:]=theta[i-1,:]+np.transpose(np.dot(Y,e))
        
        theta1[i,:]=theta[i,0:na1+nb1]
        theta2[i,:]=theta[i,na1+nb1:na1+nb1+na2+nb2]
        theta3[i,:]=theta[i,na1+nb1+na2+nb2:] 
    
        h1=np.vstack((-xp1,up1))  
        h2=np.vstack((-xp2,up2))    
        h3=np.vstack((-wp,vp))   
        h=np.vstack((h1,h2))
        h=np.vstack((h,h3))
        #t[i]=i*Ts  
        
        thetak=np.transpose(theta[i,:])
        

        ypred=np.zeros([N,1])
    #    yp=np.roll(yp,1)
    #    yp[0,0]=y1[i,0]
        thetak1=thetak[0:na1+nb1]
        thetak2=thetak[na1+nb1:na1+nb1+na2+nb2]
        thetak3=thetak[na1+nb1+na2+nb2:] 
        xp1t=xp1
        xp2t=xp2
        wpt=wp
        vpt=vp
        h1t=h1
        h2t=h2
        h3t=h3
        ht=np.vstack((h1t,h2t))
        ht=np.vstack((ht,h3t))
        for k in range(N):
            
            if k>0:
    #            yp=np.roll(yp,1)
    #            yp[0,0]=ypred[k-1] 
                xp1t=np.roll(xp1t,1)
                xp1t[0,0]=np.dot(h1t[:,0],thetak1)
                xp2t=np.roll(xp2t,1)
                xp2t[0,0]=np.dot(h2t[:,0],thetak2)
                up1=np.roll(up1,1)
                up1[0,0]=0            
                up2=np.roll(up2,1)
                up2[0,0]=0 
                
                wpt=np.roll(wpt,1)
                wpt[0,0]=ypred[k-1,0]-(np.dot(h1t[:,0],thetak1)+np.dot(h2t[:,0],thetak2))
    #            wpt[0,0]=0
                vpt=np.roll(vpt,1)
                vpt[0,0]=0                        
    #            ep=np.roll(ep,1)
    #            ep[0,0]=0                               
                h1t=np.vstack((-xp1t,up1))  
                h2t=np.vstack((-xp2t,up2))    
                h3t=np.vstack((-wpt,vpt))   
                ht=np.vstack((h1t,h2t))
                ht=np.vstack((ht,h3t))
        
            ypred[k,0]=np.dot(ht[:,0],thetak)
            
        ypredN[i,0]=ypred[-1,0]
        GpredN[i,0]=ypredN[i,0]+Gpb

    return GpredN,theta,theta1,theta2,theta3,thetak,thetak1,thetak2,thetak3,Gf
    
# prediction N-steps ahead using Box-Jenkins (1 input) and recursive least squares algorithm for parameter identification    
# y = B1/A1 * vsig + C/D * e  
def GpredBJ1(G,Gpb,vsig,na1,nb1,nc,nd,N,fz,mao):
    # G - array of CGM data
    # Gpb - glycemia operating point (estimation of basal glycemia)
    # vsig - input - rate of insulin administration
    # na1 - order of polynomial A1
    # nb1 - order of polynomial B1
    # nc - order of polynomial C
    # nd - order of polynomial D
    # N - prediction horizon
    # fz - forgetting factor (if it equals to 1 - means no forgetting)
    # mao - order of MA filter for CGM data (if mao=0 - means no filter is used)
    idx_final=G.shape[0]
    y1=np.zeros([idx_final,1])
    e1=np.zeros([idx_final,1])

    Gf=np.zeros([idx_final,1])
    Gf[0,0]=G[0]
    if mao>0:
        ypf=np.zeros([mao,1])
        ypf[0:mao,0]=G[0:mao,0]

    xp1=np.zeros([na1,1])
    up1=np.zeros([nb1,1])
    wp=np.zeros([nc,1])
    vp=np.zeros([nd,1])
    
    theta1=np.zeros([idx_final,na1+nb1])
    theta2=np.zeros([idx_final,nc+nd])
    theta=np.zeros([idx_final,na1+nb1+nc+nd])
    P=np.eye(na1+nb1+nc+nd)*1e6
    
    h1=np.vstack((-xp1,up1))  
    h2=np.vstack((-wp,vp))   
    h=np.vstack((h1,h2))
    
    ypredN=np.zeros([idx_final,1])
    GpredN=np.zeros([idx_final,1])

    for i in range(0,idx_final):    
        y1[i,0]=G[i]-Gpb

        if mao>0:
            if i>0:
                ypf=np.roll(ypf,1)
                ypf[0,0]=G[i]

                Gf[i]=1/(mao+1)*(G[i]+np.sum(ypf))
            y1[i,0]=Gf[i]-Gpb
        
    #    for c in range(1,na+1):
    #        if i-c>=0:
    #            yp[c-1,0]=y1[i-c]
    #        else:
    #            yp[c-1,0]=0
        
    #    for c in range(1,na1+1):
    #        if i-c>=0:
    #            xp1[c-1,0]=np.dot(h1[:,0],theta1[i-1,:])
    #        else:
    #            xp1[c-1,0]=0
           
        xp1=np.roll(xp1,1)
        xp1[0,0]=np.dot(h1[:,0],theta1[i-1,:])
        
    #    for c in range(1,na2+1):
    #        if i-c>=0:
    #            xp2[c-1,0]=np.dot(h2[:,0],theta2[i-1,:])
    #        else:
    #            xp2[c-1,0]=0
    
        for c in range(1,nb1+1):
            if i-c>=0:
                up1[c-1,0]=vsig[i-c]-vsig[0]
            else:
                up1[c-1,0]=0 
                    
              
    #    for c in range(1,nc+1):
    #        if i-c>=0:
    #            ep[c-1,0]=e1[i-c]
    #        else:
    #            ep[c-1,0]=0 
    
        wp=np.roll(wp,1)
        wp[0,0]=y1[i,0]-np.dot(h1[:,0],theta1[i-1,:])
        
        vp=np.roll(vp,1)
        vp[0,0]=y1[i,0]-np.dot(h[:,0],theta[i-1,:])         
        
    #    h1=np.vstack((-xp1,up1))  
    #    h2=np.vstack((-xp2,up2))    
    #    h3=np.vstack((-wp,vp))   
    #    h=np.vstack((h1,h2))
    #    h=np.vstack((h,h3))
        
    #    H[i,:]=np.transpose(h)    
        
        e=y1[i,0]-np.dot(h[:,0],theta[i-1,:])
        e1[i,0]=e
        Y=np.dot(P,h)/(1+np.dot(np.dot(np.transpose(h),P),h))
        P=(P-np.dot(np.dot(Y,np.transpose(h)),P))*1/fz
        theta[i,:]=theta[i-1,:]+np.transpose(np.dot(Y,e))
        
        theta1[i,:]=theta[i,0:na1+nb1]
        theta2[i,:]=theta[i,na1+nb1:]
    
        h1=np.vstack((-xp1,up1))  
        h2=np.vstack((-wp,vp))   
        h=np.vstack((h1,h2))
        #t[i]=i*Ts  
        
        thetak=np.transpose(theta[i,:])
        

        ypred=np.zeros([N,1])
    #    yp=np.roll(yp,1)
    #    yp[0,0]=y1[i,0]
        thetak1=thetak[0:na1+nb1]
        thetak2=thetak[na1+nb1:]
        xp1t=xp1
        wpt=wp
        vpt=vp
        h1t=h1
        h2t=h2
        ht=np.vstack((h1t,h2t))
        for k in range(N):
            
            if k>0:
    #            yp=np.roll(yp,1)
    #            yp[0,0]=ypred[k-1] 
                xp1t=np.roll(xp1t,1)
                xp1t[0,0]=np.dot(h1t[:,0],thetak1)
                up1=np.roll(up1,1)
                up1[0,0]=0            
                
                wpt=np.roll(wpt,1)
                wpt[0,0]=ypred[k-1,0]-np.dot(h1t[:,0],thetak1)
    #            wpt[0,0]=0
                vpt=np.roll(vpt,1)
                vpt[0,0]=0                        
    #            ep=np.roll(ep,1)
    #            ep[0,0]=0                               
                h1t=np.vstack((-xp1t,up1))  
                h2t=np.vstack((-wpt,vpt))   
                ht=np.vstack((h1t,h2t))
        
            ypred[k,0]=np.dot(ht[:,0],thetak)
            
        ypredN[i,0]=ypred[-1,0]
        GpredN[i,0]=ypredN[i,0]+Gpb

    return GpredN,theta,theta1,theta2,thetak,thetak1,thetak2,Gf
 
    
        