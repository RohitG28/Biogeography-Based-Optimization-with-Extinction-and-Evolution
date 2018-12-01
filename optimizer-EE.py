import BBOEE as bbo_ee
import benchmarks
import csv
import numpy as np
import time
import matplotlib.pyplot as plt


def selector(algo,func_details,popSize,Iter):
    function_name=func_details[0]
    lb=func_details[1]
    ub=func_details[2]
    dim=func_details[3]
       
    if(algo==0):
        x=bbo_ee.BBOEE(getattr(benchmarks, function_name),lb,ub,dim,popSize,Iter)    
    return x
    
    
# Select optimizers
BBO= True # Code by Raju Pal & Himanshu Mittal


# Select benchmark function
F1=True
F2=False
F3=False
F4=False
F5=False
F6=False
F7=False
F8=False
F9=False
F10=False
F11=False
F12=False
F13=False
F14=False
F15=False
F16=False
F17=False
F18=False
F19=False



optimizer=[BBO]
benchmarkfunc=[F1,F2,F3,F4,F5,F6,F7,F8,F9,F10,F11,F12,F13,F14,F15,F16,F17,F18,F19] 
        
# Select number of repetitions for each experiment. 
# To obtain meaningful statistical results, usually 30 independent runs are executed for each algorithm.
NumOfRuns=30

# Select general parameters for all optimizers (population size, number of iterations)
PopulationSize = 50
Iterations= 1000

#Export results ?
# Export=True

#Automaticly generated name by date and time
# ExportToFile="experiment"+time.strftime("%Y-%m-%d-%H-%M-%S")+".csv" 

# Check if it works at least once
Flag=False

# CSV Header for the cinvergence 
CnvgHeader=[]

for l in range(0,Iterations):
	CnvgHeader.append("Iter"+str(l+1))


for i in range (0, len(optimizer)):
    for j in range (0, len(benchmarkfunc)):
        if((optimizer[i]==True) and (benchmarkfunc[j]==True)): # start experiment if an optimizer and an objective function is selected
            sumOfRuns = 0
            for k in range (0,NumOfRuns):
                
                func_details=benchmarks.getFunctionDetails(j)
                x=selector(i,func_details,PopulationSize,Iterations)
                # if(Export==True):
                #     with open(ExportToFile, 'a') as out:
                #         writer = csv.writer(out,delimiter=',')
                #         if (Flag==False): # just one time to write the header of the CSV file
                #             header= np.concatenate([["Optimizer","objfname","startTime","EndTime","ExecutionTime"],CnvgHeader])
                #             writer.writerow(header)
                #         a=np.concatenate([[x.optimizer,x.objfname,x.startTime,x.endTime,x.executionTime],x.convergence])
                #         writer.writerow(a)
                #     out.close()
                # plt.plot(x.convergence,label=x.objfname+"/"+str(k))
                sumOfRuns += x.best
                Flag=True # at least one experiment
            print(float(sumOfRuns)/NumOfRuns)
# plt.xlabel("Iteration No.")
# plt.ylabel("Fitness")
# plt.show()
                
if (Flag==False): # Faild to run at least one experiment
    print("No Optomizer or Cost function is selected. Check lists of available optimizers and cost functions") 
        
        