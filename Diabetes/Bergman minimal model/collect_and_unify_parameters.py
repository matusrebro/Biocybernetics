import numpy as np


# --- glucose subsystem model parameters
parG1h = np.loadtxt('parG1h.csv') # normal
parG1d = np.loadtxt('parG1d.csv') # t2dm

parG1ho = np.loadtxt('parG1h3.csv')
parG1do = np.loadtxt('parG1d3.csv')



parGh = np.mean(np.vstack((parG1h, parG1ho)),0)
parGd = np.mean(np.vstack((parG1d, parG1do)),0)



parI2h=np.loadtxt('parI2h.csv')
parI2d=np.loadtxt('parI2d.csv')

parI2ho=np.loadtxt('parI2h3.csv')
parI2do=np.loadtxt('parI2d3.csv')



parIh = np.mean(np.vstack((parI2h, parI2ho)),0)
parId = np.mean(np.vstack((parI2d, parI2do)),0)



parGho = np.hstack((parGh,np.loadtxt('parG1h3o.csv')))
parGdo = np.hstack((parGd,np.loadtxt('parG1d3o.csv')))


parIh1 = np.hstack((parIh, np.loadtxt('parI2h3o0.csv')))
parIh2 = np.hstack((parIh, np.loadtxt('parI2h3o1.csv')))
parIh3 = np.hstack((parIh, np.loadtxt('parI2h3o2.csv')))
parIh4 = np.hstack((parIh, np.loadtxt('parI2h3o3.csv')))


parId1 = np.hstack((parId, np.loadtxt('parI2d3o0.csv')))
parId2 = np.hstack((parId, np.loadtxt('parI2d3o1.csv')))
parId3 = np.hstack((parId, np.loadtxt('parI2d3o2.csv')))
parId4 = np.hstack((parId, np.loadtxt('parI2d3o3.csv')))



parh1 = np.hstack((parGho, parIh1))
parh2 = np.hstack((parGho, parIh2))
parh3 = np.hstack((parGho, parIh3))
parh4 = np.hstack((parGho, parIh4))

pard1 = np.hstack((parGdo, parId1))
pard2 = np.hstack((parGdo, parId2))
pard3 = np.hstack((parGdo, parId3))
pard4 = np.hstack((parGdo, parId4))



np.savetxt('par_normal_mean_inc1.csv', parh1)
np.savetxt('par_normal_mean_inc2.csv', parh2)
np.savetxt('par_normal_mean_inc3.csv', parh3)
np.savetxt('par_normal_mean_inc4.csv', parh4)

np.savetxt('par_t2dm_mean_inc1.csv', pard1)
np.savetxt('par_t2dm_mean_inc2.csv', pard2)
np.savetxt('par_t2dm_mean_inc3.csv', pard3)
np.savetxt('par_t2dm_mean_inc4.csv', pard4)




