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

        # Infect 0.01% of the vertices, random.
        self.infected = (np.random.rand(self.N) < 0.01).astype(int)

    def step(self):
        """ Do the calculation for this time step """

        for j in xrange(0, self.N):

            # If the current node is not infected, we dont have to check this one
            if (not self.infected[j]):
                continue

            neighbors = nx.all_neighbors(self.G, j)

            # Loop through all the neighbors, and count infected
            for neighbor in neighbors:

                if (self.infected[neighbor] == 0):
                    self.checkInfected(neighbor)

        self.prevalence[self.t][self.repeat] = float(sum(self.infected))/float(self.N)

        self.t += 1

    def checkInfected(self, vertex):
        """ Check if the given vertex is infected in the next timestep """

        if (random.random() <= self.i):
            self.infected[vertex] = 1

    def getStats(self):
        """ Get back the prevalence """
        return self.prevalence


def printGraph(prevalence, tTotal, repeat, label, stepsize = 10):
    """ Add the errorbarplot for this simulation to the figure """

    y = np.empty([int(tTotal/stepsize)], dtype=float)
    x = np.arange(0, tTotal, stepsize)

    j = 0

    # calculate the averages for each step
    for i in xrange(0, tTotal):

        if (i % stepsize == 0):
            step = prevalence[i]
            y[j] = np.average(step)
            j += 1

    # plot the errorbar
    plt.errorbar(x, y, label=label)


def runSim(tTotal, repeat, N, k, i):
    """ Run the simulation with the given settings"""

    # Make an array to save the data
    prevalence = np.ndarray(shape=(tTotal, repeat), dtype=float)

    for j in xrange(0, repeat):

        graph = erdos(10**4, 0.01, 5.0, prevalence, j)

        for i in xrange(0, tTotal):
            graph.step()

        prevalence = graph.getStats()

    return prevalence

def main():

    tTotal = 100
    repeat = 10

    # Run the two given simulations
    sim1 = runSim(tTotal, repeat, 10**5, 0.01, 5.0)
    sim2 = runSim(tTotal, repeat, 10**5, 0.1, 0.8)

    plt.figure()

    printGraph(sim1, tTotal, repeat, "N=10^5, i = 0.01, <k> = 5.0", stepsize=10)
    printGraph(sim2, tTotal, repeat, "N=10^5, i = 0.1, <k> = 0.8", stepsize=10)

    plt.legend()

    plt.title("Question 1b with 2 simulations")
    plt.show()


if __name__ == "__main__":
    main()


