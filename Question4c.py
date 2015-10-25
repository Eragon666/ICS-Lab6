import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math as Math
import random

class erdos:

    def __init__(self, N, i, k, neighbors, repeat):

        self.N = N
        self.i = i
        self.k = k
        self.e = 2.5

        self.neighbors = neighbors
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
        self.vertexes = 0
        self.newinfected = 0

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

    def calcNeighbors(self):
        """ Calculate the average of edges for newly infeced nodes in this step """

        # Make sure we don't have division by zero errors
        if (self.vertexes == 0 or self.newinfected == 0):
            self.neighbors[self.t][self.repeat] = 0
        else:   
            self.neighbors[self.t][self.repeat] = self.vertexes/self.newinfected

    def checkInfected(self, vertex):
        """ Check if the given vertex is infected in the next timestep """

        if (random.random() <= self.i):
            self.infected[vertex] = 1
            
            # Count the number of neighbors of the newly infected vertex/node
            nrneighbors = len(self.G.neighbors(vertex))
            
            self.vertexes += nrneighbors
            self.newinfected += 1
        

    def getStats(self):
        """ Get back the prevalence """
        return self.neighbors


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

        for l in xrange(1, tTotal):
            graph.step(l)
            graph.calcNeighbors()

        prevalence = graph.getStats()

    return prevalence

def main():

    tTotal = 50
    repeat = 5

    # Run the two given simulations
    print 'Running simulation 1/1'
    sim1 = runSim(tTotal, repeat, 10**5, 0.01, 5.0)

    plt.figure()

    printGraph(sim1, tTotal, repeat, "Average edges newly infected", stepsize=1)

    plt.legend(loc=2)

    plt.title("Question 4c with scale-free network")

    print """ The figure shows the average number of edges for every newly infected node in a scale-free network simulation for <k>=5, i=0.1 and N=10^5. Every simulation is simulated with """ + str(tTotal) + """ steps and this is repeated. 
    """ + str(repeat) + """ times. """

    plt.show()


if __name__ == "__main__":
    main()


