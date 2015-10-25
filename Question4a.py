import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math as Math
import random

import Question1 as Erdos

class erdos:

    def __init__(self, N, i, k, prevalence, repeat):

        self.N = N
        self.i = i
        self.k = k
        self.e = 2.5

        self.prevalence = prevalence
        self.repeat = repeat

        self.t = 0

        self.createGraph()

    def createGraph(self):
        """ Calculate the basic data for the graph """
        """ Generate everything for the scale free network """

        # Create a graph with degrees following a power law distribution

        s = []

        count = 0

        while len(s) < self.N:
            nextval = int(nx.utils.powerlaw_sequence(int(self.k), self.e)[0])
            
            if nextval != 0:
                count += nextval
                s.append(nextval)
                
        # s scaled and rounded such that the average degree equals k
        s = s / np.mean(s) * self.k
        s = np.around(s).astype(int)

        # Sum of degrees must be even. I added one edge to the first node to fix this
        if sum(s) % 2:
            s[0] += 1
            
        G = nx.configuration_model(s)
        G = nx.Graph(G)
           
        # Remove self-loops
        G.remove_edges_from(G.selfloop_edges())
            
        self.G = G
        
        self.generateInfected()
        
        
    def generateInfected(self):
        """ Generate infected nodes """
        
        # Infect 0.01% of the vertices, random
        nrinfected = int(float(self.N)*float(0.001))
        infected = random.sample(xrange(0, self.N), nrinfected)

        self.infected = np.zeros((self.N,), dtype=np.int)

        for val in infected:
            self.infected[infected] = 1

    def step(self, t):
        """ Do the calculation for this time step """

        self.t = t

        # Make a copy of self.infected to make sure that only vertexes that
        # were already infected can infect other vertexes in this step.
        infected = np.copy(self.infected)

        for j in xrange(0, self.N):

            # If the current node is not infected, we dont have to check this one
            if (not self.infected[j]):
                continue

            neighbors = nx.all_neighbors(self.G, j)

            # Loop through all the neighbors, and count infected
            for neighbor in neighbors:

                if (infected[neighbor] == 0):
                    self.checkInfected(neighbor)

    def calcPrevalence(self):
        """ Calculate the prevalence """

        self.prevalence[self.t][self.repeat] = sum(self.infected)/float(self.N)

    def checkInfected(self, vertex):
        """ Check if the given vertex is infected in the next timestep """

        if (random.random() <= self.i):
            self.infected[vertex] = 1

    def getStats(self):
        """ Get back the prevalence """
        return self.prevalence


def printGraph(prevalence, tTotal, repeat, label, stepsize = 10):
    """ Add the errorbarplot for this simulation to the figure """

    x = np.arange(0, tTotal, stepsize)
    y = np.empty([int(tTotal/stepsize)], dtype=float)
    std = np.empty([int(tTotal/stepsize)], dtype=float)

    j = 0

    # calculate the averages for each step
    for i in xrange(0, tTotal):

        if (i % stepsize == 0):
            step = prevalence[i]
            y[j] = np.average(step)

            # I enabled this print and set the tTotal to 2 to get the answer
            # for 1e. 
            #print y[j]

            std[j] = np.std(step)
            j += 1

    # plot the errorbar
    plt.errorbar(x, y, yerr=std, label=label)


def runSim(tTotal, repeat, N, i, k):
    """ Run the simulation with the given settings"""

    # Make an array to save the data
    prevalence = np.ndarray(shape=(tTotal, repeat), dtype=float)

    for j in xrange(0, repeat):

        print str(j+1) + '/' + str(repeat)

        graph = erdos(N, i, k, prevalence, j)

        # Save the prevalence of step 0
        graph.calcPrevalence()

        for l in xrange(1, tTotal):
            graph.step(l)
            graph.calcPrevalence()

        prevalence = graph.getStats()

    return prevalence

def main():

    tTotal = 350
    repeat = 25

    # Run the two given simulations
    print 'Running simulation 1/2'
    sim1 = runSim(tTotal, repeat, 10**5, 0.01, 5.0)
    print 'Running simulation 2/2'
    sim2 = Erdos.runSim(tTotal, repeat, 10**5, 0.01, 5.0)

    plt.figure()

    printGraph(sim1, tTotal, repeat, "Scale-free network", stepsize=1)
    printGraph(sim2, tTotal, repeat, "Erdos-Renyi network", stepsize=1)

    plt.legend(loc=2)

    plt.title("Question 4a with 2 simulations")

    print """ The figure shows the Erdos-Renyi and scale-free network simulation for <k>=5, i=0.1 and N=10^5. Every simulation is simulated with """ + str(tTotal) + """ steps and this is repeated
    """ + str(repeat) + """ times. """

    plt.show()


if __name__ == "__main__":
    main()


