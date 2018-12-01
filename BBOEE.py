from __future__ import division
import numpy as np
import random
import math
import time
 
class solution:
    def __init__(self):
        self.best = 0
        self.bestIndividual=[]
        self.convergence = []
        self.optimizer=""
        self.objfname=""
        self.startTime=0
        self.endTime=0
        self.executionTime=0
        self.lb=0
        self.ub=0
        self.dim=0
        self.popnum=0
        self.maxiers=0

def ClearDups(Population, PopSize, dim, MaxParValue, MinParValue):

    for i in range(PopSize):
        Chrom1 = np.sort(Population[i,:]);
        for j in range(i+1,PopSize):
            Chrom2 = np.sort(Population[j,:]);
            if Chrom1 is Chrom2:
                parnum = np.ceil(dim * random.random());
                Population[j,parnum] = MinParValue + (MaxParValue - MinParValue) * random.random();
    return Population
  
def generateRandomNums(populationSize,k):
    r1 = random.randint(0,populationSize-1)
    while r1 == k:
        r1 = random.randint(0,populationSize-1)

    r2 = random.randint(0,populationSize-1)
    while r2 == k or r2 == r1:
        r2 = random.randint(0,populationSize-1)

    r3 = random.randint(0,populationSize-1)
    while r3 == k or r3 == r1 or r3 == r2:
        r3 = random.randint(0,populationSize-1)

    return r1,r2,r3


def BBOEE(objf,lb,ub,noOfDimensions,populationSize,noOfIterations):
    # Defining the solution variable for saving output variables
    s=solution()
    
    # Initializing BBO parameters
    pmutate = 0; # initial mutation probability
    Keep = 2; # elitism parameter: how many of the best habitats to keep from one generation to the next
    CR = 0.5 # crossover rate 
    F = 0.5
    c = 50
    partition = 20

    # Initializing the parameters with default values
    fit = np.zeros(populationSize)
    EliteSolution=np.zeros((Keep,noOfDimensions))
    EliteCost=np.zeros(Keep)
    Island=np.zeros((populationSize,noOfDimensions))
    mu=np.zeros(populationSize)
    lambda1=np.zeros(populationSize)
    MinCost=np.zeros(noOfIterations)
    Bestpos=np.zeros(noOfDimensions)


    # Initializing Population
    population=np.random.uniform(0,1,(populationSize,noOfDimensions)) *(ub-lb)+lb
    
    #Calculate objective function for each particle
    for i in range(populationSize):
        # Performing the bound checking
        population[i,:]=np.clip(population[i,:], lb, ub)
        fitness=objf(population[i,:])
        fit[i]=fitness

    # Calculating the mu and lambda1
    ########################################################################### muMax, lambda1Max, calculation should be based on fitness
    for i in range(populationSize):
        mu[i] = (1 - ((i+1)/populationSize)) + c/(i+1);
        lambda1[i] = ((i+1)/populationSize)*((i+1)*(populationSize-(i+1))/populationSize**2) + 2*((populationSize*(i+1))**0.5);

    # for i in range(populationSize):
    #     mu[i] = (populationSize + 1 - (i)) / (populationSize + 1)
    #     lambda1[i] = 1 - mu[i]

    print("BBO is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")

    # Sort the population on fitness
    I=np.argsort(fit)
    population=population[I,:]

    # Defining the loop
    for l in range(noOfIterations):
        selected = 0
        selectedList = []
        while selected < partition:
            index = np.random.randint(0,populationSize-1)
            if index not in selectedList:
                selectedList.append(index)
                selected = selected+1

        # Defining the Elite Solutions
        for j in range(Keep):
            EliteSolution[j,:]=population[j,:]
            EliteCost[j]=fit[j]

        # Performing Migration operator on Group A 
        for k in range(populationSize):
            if k not in selectedList:
                # r1,r2,r3 = generateRandomNums(populationSize,k)
                # jRandom = random.randint(0,noOfDimensions)
                for j in range(noOfDimensions):
                    if random.random() < lambda1[k]:
                        RandomNum = random.random() * sum(mu);
                        Select = mu[0];
                        SelectIndex = 0;
                        while (RandomNum > Select) and (SelectIndex < (populationSize-1)):
                            SelectIndex = SelectIndex + 1;
                            Select = Select + mu[SelectIndex];
                        r = random.randint(0,populationSize-1)
                        while r==k or r==SelectIndex:
                            r = random.randint(0,populationSize-1)

                        Island[k,j] = population[SelectIndex,j] + random.uniform(-1,1)*(population[SelectIndex,j] - population[r,j])
                    else:
                        Island[k,j] = population[k,j]

        # Performing Migration operator on Group B
        random.shuffle(selectedList)
        for k in range(len(selectedList)):
            for j in range(noOfDimensions):
                if random.random() < lambda1[k]:
                # if j == jRandom:
                #     Island[k,j] = population[r1,j] + random.random()*(population[r2,j]-population[r3,j])
                # else:
                # Performing Roulette Wheel
                ###############################################################################
                    if k == 0:
                        previous = len(selectedList)-1
                        next1 = k+1
                    elif k == len(selectedList)-1:
                        previous = k-1
                        next1 = 0
                    else: 
                        previous = k-1
                        next1 = k+1

                    if mu[selectedList[previous]] > mu[selectedList[next1]]:
                        SelectIndex = selectedList[previous]
                    else: 
                        SelectIndex = selectedList[next1]

                    if random.random() < mu[SelectIndex]:
                        r = random.randint(0,populationSize-1)

                        while r==k or r==SelectIndex:
                            r = random.randint(0,populationSize-1)

                        Island[k,j] = population[SelectIndex,j] + random.uniform(-1,1)*(population[SelectIndex,j] - population[r,j])
                    else:
                        Island[k,j] = population[k,j]
                else:
                    Island[k,j] = population[k,j]


        # Performing the bound checking
        for i in range(populationSize):
            Island[i,:]=np.clip(Island[i,:], lb, ub)

        # Replace the habitats with their new versions.
        for k in range(populationSize):
            population[k,:] = Island[k,:]

        #Calculate objective function for each individual
        for i in range(populationSize):
            fitness=objf(population[i,:])
            fit[i]=fitness

        # Sort the fitness
        fitness_sorted=np.sort(fit)
        # fitness_sorted=fit[::-1].sort()

        # Sort the population on fitness
        I=np.argsort(fit)
        population=population[I,:]

        # Replacing the individual of population with EliteSolution
        for k in range(Keep):
            population[(populationSize-1)-k,:] = EliteSolution[k,:];
            fit[(populationSize-1)-k] = EliteCost[k];
        
        # Removing the duplicate individuals
        ########################################################### necessary??????????????????
        population=ClearDups(population, populationSize, noOfDimensions, ub, lb)

        #Calculate objective function for each individual
        for i in range(populationSize):
            fitness=objf(population[i,:])
            fit[i]=fitness

        # Sort the fitness
        fitness_sorted=np.sort(fit)
        #fitness_sorted = fit[::-1].sort()

        # Sort the population on fitness
        I=np.argsort(fit)  
        population=population[I,:]

        # Saving the best individual
        MinCost[l] = fit[0]
        Bestpos=population[0,:]
        gBestScore=fit[0]
        s.best = gBestScore

        normalizedFit = fit/np.sum(fit)
        HSI = normalizedFit
        HSI = np.floor(populationSize*HSI)

        # for i in range(populationSize):
        #     mu[i] = (1 - ((HSI[i]+1)/populationSize)) + c/(HSI[i]+1);
        #     lambda1[i] = ((HSI[i]+1)/populationSize)*((HSI[i]+1)*(populationSize-(HSI[i]+1))/populationSize**2) + 2*((populationSize*(HSI[i]+1))**0.5);

        # Displaying the best fitness of each iteration
        print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(gBestScore)]);

    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=MinCost
    s.optimizer="BBO"
    s.objfname=objf.__name__

    return s