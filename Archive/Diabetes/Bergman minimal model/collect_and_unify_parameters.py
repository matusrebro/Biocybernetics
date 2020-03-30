import numpy as np


# --- glucose subsystem model parameters

# intravenous inputs
parG1h = np.loadtxt('parG1h.csv') # normal
parG1o = np.loadtxt('parG1o.csv') # obese
parG1d = np.loadtxt('parG1d.csv') # t2dm


# oral inputs
parG1ho = np.loadtxt('parG1h3.csv') # normal
parG1do = np.loadtxt('parG1d3.csv') # t2dm


# --- insulin subsystem model parameters

# intravenous inputs
parI2h = np.loadtxt('parI2h.csv') # normal
parI2o = np.loadtxt('parI2o.csv') # obese
parI2d = np.loadtxt('parI2d.csv') # t2dm 

# oral inputs
parI2ho = np.loadtxt('parI2h3.csv') # normal
parI2do = np.loadtxt('parI2d3.csv') # t2dm




# --- glucose subsystem + glucose absorption subsystem (oral inputs)
parGho = np.hstack((parG1ho,np.loadtxt('parG1h3o.csv'))) # normal
parGdo = np.hstack((parG1do,np.loadtxt('parG1d3o.csv'))) # t2dm


# --- insulin subsystem + incretin effect subsystems (oral inputs)

# normal subjects
parIh1 = np.hstack((parI2ho, np.loadtxt('parI2h3o0.csv'))) # model 1
parIh2 = np.hstack((parI2ho, np.loadtxt('parI2h3o1.csv'))) # model 2
parIh3 = np.hstack((parI2ho, np.loadtxt('parI2h3o2.csv'))) # model 3
parIh4 = np.hstack((parI2ho, np.loadtxt('parI2h3o3.csv'))) # model 4

# t2dm subjects
parId1 = np.hstack((parI2do, np.loadtxt('parI2d3o0.csv'))) # model 1
parId2 = np.hstack((parI2do, np.loadtxt('parI2d3o1.csv'))) # model 2
parId3 = np.hstack((parI2do, np.loadtxt('parI2d3o2.csv'))) # model 3
parId4 = np.hstack((parI2do, np.loadtxt('parI2d3o3.csv'))) # model 4


# --- unification of parameters

# intravenous inputs models:
parhi = np.hstack((parG1h, parI2h)) # normal
paroi = np.hstack((parG1o, parI2o)) # obese
pardi = np.hstack((parG1d, parI2d)) # t2dm


# oral inputs models:
parho1 = np.hstack((parGho, parIh1))
parho2 = np.hstack((parGho, parIh2))
parho3 = np.hstack((parGho, parIh3))
parho4 = np.hstack((parGho, parIh4))

pardo1 = np.hstack((parGdo, parId1))
pardo2 = np.hstack((parGdo, parId2))
pardo3 = np.hstack((parGdo, parId3))
pardo4 = np.hstack((parGdo, parId4))


np.savetxt('par_normal_iv.csv', parhi)
np.savetxt('par_obese_iv.csv', paroi)
np.savetxt('par_t2dm_iv.csv', pardi)

np.savetxt('par_normal_oral_inc1.csv', parho1)
np.savetxt('par_normal_oral_inc2.csv', parho2)
np.savetxt('par_normal_oral_inc3.csv', parho3)
np.savetxt('par_normal_oral_inc4.csv', parho4)

np.savetxt('par_t2dm_oral_inc1.csv', pardo1)
np.savetxt('par_t2dm_oral_inc2.csv', pardo2)
np.savetxt('par_t2dm_oral_inc3.csv', pardo3)
np.savetxt('par_t2dm_oral_inc4.csv', pardo4)




