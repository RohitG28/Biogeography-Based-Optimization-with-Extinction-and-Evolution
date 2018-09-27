import numpy as np
import math

# define the function blocks
    
def F1(x):
    s=np.sum(x**2);
    return s

#Ackley function
def F2(x):
	firstSum = 0.0
	secondSum = 0.0
	for c in x:
		firstSum += c**2.0
		secondSum += math.cos(2.0*math.pi*c)
	n = float(len(x))
	return -20.0*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e

#Griewank function
def F3(x):
	part1 = 0
	for i in range(len(x)):
		part1 += x[i]**2
		part2 = 1
	for i in range(len(x)):
		part2 *= math.cos(float(x[i]) / math.sqrt(i+1))
	return 1 + (float(part1)/4000.0) - float(part2)

#Rastrigin function
def F4(x):
	fitness = 10*len(x)
	for i in range(len(x)):
		fitness += x[i]**2 - (10*math.cos(2*math.pi*x[i]))
	return fitness

def getFunctionDetails(a):
    
    # [name, lb, ub, dim]
    param = {	
    			0: ["F1",-100,100,30],
    			1: ["F2",-32,32,30],
    			2: ["F3",-100,100,30],
    			3: ["F4",-100,100,30] 
            }
    return param.get(a, "nothing")



