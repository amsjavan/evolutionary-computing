import copy
import random
from math import ceil
import matplotlib.pyplot as plt
import time
import numpy
from individual import Individual
import problem


def crossover(parent1, parent2, cross_point=None):
    size = len(parent1)
    if cross_point is None:
        cross_point = random.randint(1, size - 1)
    child1 = parent1[:cross_point]
    child1.extend(parent2[cross_point:])

    child2 = parent2[:cross_point]
    child2.extend(parent1[cross_point:])

    return child1, child2


# Change the chromosome object instead of return new object
def mutate(chromosome, mu=0.0001):
    size = len(chromosome)
    ngen = ceil(mu * size)
    for _ in range(ngen):
        mutation_point = random.randint(1, size - 1)
        chromosome[mutation_point] = 1 - chromosome[mutation_point]


# genotype is list and return integer
def get_phenotype(genotype):
    return int(''.join(str(c) for c in genotype), 2)


# phenotype is integer and return list
def get_genotype(phenotype):
    genotype = '{0:032b}'.format(phenotype)
    return list(int(c) for c in genotype)


def simple_cost(self, x):
    return self.get_phenotype(x) ** 2


class GeneticAlgorithm(object):
    def simple_cost(self, x):
        return self.get_phenotype(x) ** 2

    def __init__(self, cost_function=None):
        if cost_function is None:
            self.cost_function = self.simple_cost
        else:
            self.cost_function = cost_function

        self.upper_bound = 213234
        self.lower_bound = 0

        # Initialization
        self.max_iteration = 100
        self.npop = 16

    def run(self):
        pop = []
        for i in range(self.npop):
            pos = self.get_genotype(random.randint(self.lower_bound, self.upper_bound))
            pop.append(Individual(pos, self.cost_function(pos)))

        pop = sorted(pop, key=lambda x: x.cost)

        best_sol = pop[0]

        pc = 0.5  # probability of population for crossover( number of parent for crossover)
        nc = 2 * round(pc * self.npop / 2)  # The number of parent must be even

        # pm = 0.3
        # nm = round(pm * npop)

        # Initialize plot
        plt.figure(1)
        plt.title('GA Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.ion()

        best_costs = []
        # Main Loop
        for it in range(self.max_iteration):

            parents = pop[:nc]
            children = []
            # iterating over every two element
            for _ in range(nc):
                # select parents randomly
                p1 = parents[random.randint(0, len(parents) - 1)]
                p2 = parents[random.randint(0, len(parents) - 1)]

                pair_child_pos = self.crossover(p1.pos, p2.pos)
                new_child1 = Individual(pair_child_pos[0])
                new_child2 = Individual(pair_child_pos[1])
                children.append(new_child1)
                children.append(new_child2)

            # mutation over children
            for child in children:
                self.mutate(child.pos)

            # merge children with parents and make new population
            parents.extend(children)

            # calculate costs
            for p in parents:
                p.cost = self.cost_function(p.pos)  # calculate new cost

            pop = sorted(parents, key=lambda x: x.cost)

            # copy to prevent from editing bestPop in the loop
            best_sol = copy.copy(min(min(pop, key=lambda x: x.cost), best_sol, key=lambda x: x.cost))
            print('best answer in iter {0} = '.format(it), self.get_phenotype(best_sol.pos))
            best_costs.append(best_sol.cost)

            # draw chart result
            # plt.scatter(it, best_sol.cost)
            # plt.pause(0.01)

            # Result

        plt.plot(best_costs, linewidth=3)
        plt.show()


class LittleGeneticAlgorithm(object):
    def simple_cost(self, x):
        return self.get_phenotype(x) ** 2

    def __init__(self, cost_function=None):
        if cost_function is None:
            self.cost_function = self.simple_cost
        else:
            self.cost_function = cost_function

        self.upper_bound = 213234
        self.lower_bound = 0

        self.min_var = -10
        self.max_var = 10

        # Initialization
        self.max_iteration = 100
        self.npop = 16

    def run(self):
        pop = []
        for i in range(self.npop):
            pos = self.get_genotype(random.randint(self.lower_bound, self.upper_bound))
            pop.append(Individual(pos, self.cost_function(pos)))

        pop = sorted(pop, key=lambda x: x.cost)

        best_sol = pop[0]

        pc = 0.5  # probability of population for crossover( number of parent for crossover)
        nc = 2 * round(pc * self.npop / 2)  # The number of parent must be even

        # pm = 0.3
        # nm = round(pm * npop)

        # Initialize plot
        plt.figure(1)
        plt.title('GA Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.ion()

        best_costs = []
        # Main Loop
        for it in range(self.max_iteration):

            parents = pop[:nc]
            children = []
            # iterating over every two element
            for _ in range(nc):
                # select parents randomly
                p1 = parents[random.randint(0, len(parents) - 1)]
                p2 = parents[random.randint(0, len(parents) - 1)]

                pair_child_pos = self.crossover(p1.pos, p2.pos)
                new_child1 = Individual(pair_child_pos[0])
                new_child2 = Individual(pair_child_pos[1])
                children.append(new_child1)
                children.append(new_child2)

            # mutation over children
            for child in children:
                self.mutate(child.pos)

            # merge children with parents and make new population
            parents.extend(children)

            # calculate costs
            for p in parents:
                p.cost = self.cost_function(p.pos)  # calculate new cost

            pop = sorted(parents, key=lambda x: x.cost)

            # copy to prevent from editing bestPop in the loop
            best_sol = copy.copy(min(min(pop, key=lambda x: x.cost), best_sol, key=lambda x: x.cost))
            print('best answer in iter {0} = '.format(it), self.get_phenotype(best_sol.pos))
            best_costs.append(best_sol.cost)

            # draw chart result
            # plt.scatter(it, best_sol.cost)
            # plt.pause(0.01)

            # Result

        plt.plot(best_costs, linewidth=3)
        plt.show()


class BinaryGeneticAlgorithm(object):
    def __init__(self, cost_function):
        self.cost_function = cost_function
        self.max_iteration = 50
        self.nvar = 60
        self.npop = 30

        self.pc = 0.7  # probability of population for crossover( number of parent for crossover)

        self.pm = 0.3
        self.mu = 0.01  # mutation rate per chromosome


    def run(self):
        # Initialize population
        pop = []
        for i in range(self.npop):
            pos = [random.randint(0, 1) for _ in range(self.nvar)]
            pop.append(Individual(pos, self.cost_function(pos)))

        pop = sorted(pop, key=lambda x: x.cost)
        best_sol = pop[0]

        # Initialize plot
        plt.figure(1)
        plt.title('GA Optimization')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.ion()

        best_costs = []
        # Main Loop
        for it in range(self.max_iteration):

            # ------crossover------
            nc = 2 * round(self.pc * self.npop / 2)  # The number of parent must be even
            cpop = pop[:nc]
            children = []
            # iterating over every two element
            for _ in range(nc):
                # select parents randomly
                p1 = cpop[random.randint(0, len(cpop) - 1)]
                p2 = cpop[random.randint(0, len(cpop) - 1)]

                pair_child_pos = crossover(p1.pos, p2.pos)
                new_child1 = Individual(pair_child_pos[0], self.cost_function(pair_child_pos[0]))
                new_child2 = Individual(pair_child_pos[1], self.cost_function(pair_child_pos[1]))
                children.append(new_child1)
                children.append(new_child2)

            # --------mutation-------
            nm = round(self.pm * self.npop)  # number of mutation
            mpop = pop[:nm]
            # mutation over children
            for parent in mpop:
                mutate(parent.pos, self.mu)
                parent.cost = self.cost_function(parent.pos)

            # merge and sort
            pop = pop + mpop + children
            pop = sorted(pop, key=lambda x: x.cost)

            # copy to prevent from editing bestPop in the loop
            # best_sol = copy.copy(min(min(pop, key=lambda x: x.cost), best_sol, key=lambda x: x.cost))
            best_sol = pop[0]
            best_costs.append(best_sol.cost)

            # # draw chart result
            # plt.scatter(it, best_sol.cost)
            # plt.pause(0.01)

            # Result
            print('best cost in iter {0} : best cost = {1} , best pos = {2}'.format(it, best_sol.cost, best_sol.pos))

        # problem.show_hub(best_sol.pos)
        plt.plot(best_costs, linewidth=3)
        plt.show()


def unitTest():
    # Test
    p1 = '01101001'
    p2 = '11001011'
    c1 = '01101011'
    c2 = '11001001'
    point = 3  # crossover point
    assert crossover(list(p1), list(p2), point) == (list(c1), list(c2))
    p = [1, 1, 0, 1, 1, 1, 0]
    mutate(p)
    print(p)


def test():
    l = [1, 2, 3, 4]
    for i in range(1000):
        y = numpy.random.random()
        plt.scatter(i, y)
        plt.pause(0.1)


myga = BinaryGeneticAlgorithm(problem.min_one)
# unitTest()
myga.run()
# test()
