from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import Question1 as Erdos

class euler:

    def __init__(self):
        plt.xlabel('t')
        plt.ylabel('y')
        plt.title('Question 3a')

    def calc(self, f, N, averageK, initial, risk, tStart=0, steps=300, stepsize=1, name='Functie'):
        """ Do the general calculations for the functions """
        
        self.steps = steps
        self.b = 1 - ((1 - risk) **(averageK/N))

        iMax = steps/stepsize + 1
        i = 0
        
        # Calculate the infected at the start
        infected = initial * N

        # Initialize the arrays
        yArray = np.ndarray((iMax,), float)
        tArray = np.ndarray((iMax,), float)

        t = tStart
        
        while t <= steps:
            
            yArray[i] = (infected/N) #prevalence
            tArray[i] = t
            t += stepsize        
            
            infected += f(t, infected, N, averageK, risk)
            i += 1

        # Plot the arrays
        plt.plot(tArray, yArray, label=name)

    def showplot(self):
        """ Print the legend and the plot """
        plt.legend(loc=2)
        
        #plt.axis((0, self.steps, -0.1, 1.1))
        
        plt.show()

    def question1(self, t, infected, N, averageK, risk):
        """ calculate the newly infected for the formula from Question 1 """
        
        result = (1-(1-risk)**((averageK/N)*infected)) * (N - infected)
        
        # We cant have a partly infected human, so round it to the nearest integer
        return round(result)

    def coupledODE(self,t, infected, N, averageK, risk):
        """ Calculate the newly infected for the coupled ODE's """
        
        S = float(N) - infected
        
        result = (1-(1-self.b)**infected) * S
        
        return result
        
def main():
    coupled = euler()
    
    tTotal = 15

    # Run the coupled ODE functions
    coupled.calc(f=coupled.coupledODE, N=10**5, averageK=5.0, initial=0.001, risk=0.01, steps=tTotal, name='Question 1 <k>=5.0 & i=0.01')
    coupled.calc(f=coupled.coupledODE, N=10**5, averageK=.8, initial=0.001, risk=0.1, steps=tTotal, name='Question 1 <k>=0.8 & i=0.1')
    
    # Run the erdos simulation
    
    repeat = 30
    
    print 'Running simulation 1/2'
    sim1 = Erdos.runSim(tTotal, repeat, 10**5, 0.01, 5.0)
    print 'Running simulation 2/2'
    sim2 = Erdos.runSim(tTotal, repeat, 10**5, 0.1, 0.8)
    
    Erdos.printGraph(sim1, tTotal, repeat, "N=10^5, i = 0.01, <k> = 5.0", stepsize=1)
    Erdos.printGraph(sim2, tTotal, repeat, "N=10^5, i = 0.1, <k> = 0.8", stepsize=1)
    
    print """ Alle methoden zijn uitgevoerd met de gegeven waarden uit de opgaves. De Erdos methodes is 
    """ + str(repeat) + """ keer uitgevoerd voor een beter gemiddeld resultaat. Verder zijn nu voor alle methodes 
    """ + str(tTotal) + """ stappen berekend. """

    coupled.showplot()

if __name__ == "__main__":
    main()
