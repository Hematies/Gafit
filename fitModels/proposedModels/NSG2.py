import array
import random

import numpy

from deap import algorithms
from deap import base
from deap import benchmarks
from deap.benchmarks.tools import diversity, convergence, hypervolume
from deap import creator
from deap import tools



# Code adapted from example:
# https://github.com/DEAP/deap/blob/master/examples/ga/nsga2.py


class NSGA2:
    def __init__(self, numGens,
                 fitnessFunction,
                 initIndividualFunction,
                 boundLow = 0.0, boundUp = 1.0,
                 numGenerations=100, numIndividuals=120,
                 crossoverRate = 0.75,
                 genMutationRate = 0.1,
                 childParentSimilarity=1.0):

        toolbox = base.Toolbox()
        self.NGEN = numGenerations
        self.MU = numIndividuals
        self.CXPB = crossoverRate

        NDIM = numGens

        creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMin)

        #toolbox.register("attr_float", uniform, boundLow, boundUp, NDIM)
        toolbox.register("individual", tools.initIterate, creator.Individual, initIndividualFunction)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", fitnessFunction)

        # Bounded crossover:
        toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=boundLow, up=boundUp, eta=childParentSimilarity)

        toolbox.register("mutate", tools.mutPolynomialBounded, low=boundLow, up=boundUp, eta=childParentSimilarity,
                         indpb=genMutationRate)


        toolbox.register("select", tools.selNSGA2)

        self.toolbox = toolbox

    # Función que ejecuta el algoritmo NSGA2:
    def fit(self, seed=None):
        random.seed(seed)
        toolbox = self.toolbox

        NGEN = self.NGEN
        MU = self.MU
        #CXPB = 0.75
        # CXPB = 0.7
        # CXPB = 0.4
        # PROBABILIDAD_MUTACION_INICIAL = 0.9
        # PROBABILIDAD_MUTACION = PROBABILIDAD_MUTACION_INICIAL

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", numpy.min, axis=0)
        stats.register("max", numpy.max, axis=0)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        pop = toolbox.population(n=MU)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)

        # Begin the generational process
        for gen in range(1, NGEN):
            # Vary the population
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.CXPB:
                    toolbox.mate(ind1, ind2)

                '''
                if random.random() <= PROBABILIDAD_MUTACION:
                    toolbox.mutate(ind1)
                    toolbox.mutate(ind2)
                '''
                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

                del ind1.fitness.values, ind2.fitness.values

            # PROBABILIDAD_MUTACION = PROBABILIDAD_MUTACION / 1.025

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = toolbox.select(pop + offspring, MU)
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)

            # Esto es para imprimir por pantalla:
            print(logbook.stream)


        print("Final population hypervolume is %f" % hypervolume(pop, [11.0, 11.0]))

        return pop, logbook