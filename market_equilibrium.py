

import numpy as np
import pandas as pd
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.optimize import fmin_powell


# In[46]:


#Define the parameters for each firm
parameters=np.matrix([[10,5,1.2],[8,5,1.1],[6,5,1],[4,5,0.9],[2,5,0.8]])

#Define a table to output the iterations information
result_table=pd.DataFrame(columns=["q1", "q2", "q3", "q4", "q5"])

#Select initial point and uncertainty measure
qk=np.matrix([[10],[10],[10],[10],[10]]) 
result_table=result_table.append({"q1":float(qk[0]),"q2":float(qk[1]),"q3":float(qk[2]),"q4":float(qk[3]),"q5":float(qk[4])}, ignore_index=True)

len_uncert=0.005
termination=False
k=0
N= len(qk)

while not termination: 
    
    sol=[]
    
    for i in range(N):
        
        #Compute the total supply without the value of the ith player
        Qk=0
        for j in range(N): 
            if j !=i: 
                Qk+=float(qk[j])
        
        # Set the values of the parameters
        ci=parameters[i,0]
        Li=parameters[i,1]
        Bi=parameters[i,2]

        #Solve the optimization problem 
        fun=lambda qi: (qi*(5000**(1/1.1)*(qi+Qk)**(-1/1.1))-ci*qi-(Bi/(Bi+1))*(Li**(-1/Bi))*(qi**((Bi+1)/Bi)))*-1
        #bnds = [(0, None)]
        #res= minimize(fun, 100, method='SLSQP', bounds=bnds)
        res=fmin_powell(fun,40)
        
        #Save the solution
        sol.append(float(res))

        
    result_table=result_table.append({"q1":np.round(sol[0],2),"q2":np.round(sol[1],2),"q3":np.round(sol[2],2),"q4":np.round(sol[3],2),"q5":np.round(sol[4],2)}, ignore_index=True)
    
    #Set the value of the new solution
    qk1=np.matrix([[sol[0]],[sol[1]],[sol[2]],[sol[3]],[sol[4]]]) 
    
    #Check termination criteria
    if LA.norm(qk1-qk)<=len_uncert: 
        termination=True
    
    #Update solution
    k+=1
    qk=qk1


# In[45]:


result_table

