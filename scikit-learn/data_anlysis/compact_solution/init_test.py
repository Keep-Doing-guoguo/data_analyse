import numpy as np
data = np.arange(0,81).reshape(9,9)
print(data)
print(data[:,1])
print(np.where(data[:,1] < 10 , True,False))
print(np.where(data[:,1] > 64,True,np.where(data[:,1] < 10 , True,False)))

print(np.where(data[:,1] < 10 , True,False))
print(np.where(data[:,1] > 64,64,np.where(data[:,1] < 10 , 10,data[:,1])))
