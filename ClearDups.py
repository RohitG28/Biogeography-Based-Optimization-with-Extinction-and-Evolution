import numpy
import random

def ClearDups(Population, PopSize, dim, MaxParValue, MinParValue):

    for i in range(PopSize):
        Chrom1 = numpy.sort(Population[i,:]);
        for j in range(i+1,PopSize):
            Chrom2 = numpy.sort(Population[j,:]);
            if Chrom1 is Chrom2:
                parnum = numpy.ceil(dim * random.random());
                Population[j,parnum] = MinParValue + (MaxParValue - MinParValue) * random.random();
    return Population
                