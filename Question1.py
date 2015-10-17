import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math as Math
import random

class erdos:

    def __init__(self, N, i, k, prevalence, repeat):

        self.N = N
        self.i = i
        self.k = k

        self.prevalence = prevalence
        self.repeat = repeat

        self.t = 0

        self.createGraph()

    def createGraph(self):
        """ Calculate the basic data for the graph """

        # Create a graph with N = self.N and <k> = self.k
        self.G = nx.fast_gnp_random_graph(self.N, self.k / self.N)

        # Infect 0.01% of the vertices, random
        nrinfected = int(float(self.N)*float(0.001))
        infected = random.sample(xrange(0, self.N), nrinfected)

        self.infected = np.zeros((self.N,), dtype=np.int)

        for val in infected:
            self.infected[infected] = 1

        # This function resulted in not exactly 0.01% of infected vertexes
        # self.infected = (np.random.rand(self.N) <= 0.001).astype(int)

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

    tTotal = 40
    repeat = 50

    # Run the two given simulations
    print 'Running simulation 1/2'
    sim1 = runSim(tTotal, repeat, 10**5, 0.01, 5.0)
    print 'Running simulation 2/2'
    sim2 = runSim(tTotal, repeat, 10**5, 0.1, 0.8)

    plt.figure()

    printGraph(sim1, tTotal, repeat, "N=10^5, i = 0.01, <k> = 5.0", stepsize=1)
    printGraph(sim2, tTotal, repeat, "N=10^5, i = 0.1, <k> = 0.8", stepsize=1)

    plt.legend()

    plt.title("Question 1b with 2 simulations")

    print """ The figure shows the two simulations needed for question 2. Every
    simulation is simulated with """ + str(tTotal) + """ steps and this is repeated
    """ + str(repeat) + """ times. """

    plt.show()


if __name__ == "__main__":
    main()


