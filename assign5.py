"""
For this assignment there is no automated testing. You will instead submit
your *.py file in Canvas. I will download and test your program from Canvas.
ref: https://arxiv.org/pdf/1203.3097.pdf used this journal article to make decision to use ox, but have since read other journal articles that disagree.
decided to stick with ox anyways.
"""

import time
import sys
import random 
from random import sample
INF = sys.maxsize
import matplotlib.pyplot as plt


def adjMatFromFile(filename):
    """ Create an adj/weight matrix from a file with verts, neighbors, and weights."""
    f = open(filename, "r")
    n_verts = int(f.readline())
    print(f" n_verts = {n_verts}")
    adjmat = [[None] * n_verts for i in range(n_verts)]
    for i in range(n_verts):
        adjmat[i][i] = 0
    for line in f:
        int_list = [int(i) for i in line.split()]
        vert = int_list.pop(0)
        assert len(int_list) % 2 == 0
        n_neighbors = len(int_list) // 2
        neighbors = [int_list[n] for n in range(0, len(int_list), 2)]
        distances = [int_list[d] for d in range(1, len(int_list), 2)]
        for i in range(n_neighbors):
            adjmat[vert][neighbors[i]] = distances[i]
    f.close()
    return adjmat

def generate_path(path_length: int):
    """ Generate a random path of ints 0 to path length and return path list. """
    path = [i for i in range(path_length)]
    random.shuffle(path)
    return path

def generate_child_path(parent1, parent2):
    """ Use OX to take 2 parent paths(list[int]) and return child path(list[int]). """
    if len(parent1) < 3:
        print("not long enough path to crossover")
        return
    if len(parent1) != len(parent2):
        print("not the same length")
        return
    
    crossover_point1 = len(parent1) // 3
    crossover_point2 = crossover_point1 * 2
    #crossover_point2 = len(parent1) - crossover_point1
    child = [None] * len(parent1)
    dummy_parent = parent2.copy()
    
    for i in range(crossover_point1, crossover_point2):
        child[i] = parent1[i]
        dummy_parent.remove(child[i])
    
    dummy_parent = dummy_parent[crossover_point1:] + dummy_parent[:crossover_point1]
    
    for j in range(crossover_point2, len(child)):
        child[j] = dummy_parent.pop(0)
    
    for j in range(crossover_point1):
        child[j] = dummy_parent.pop(0)

    return child
    
def mutation(path, mutation_rate):
    """ Randomly switch values at indices of a list at a rate of mutation_rate and return the list. """
    for i in range(len(path)//2):
        if random.random() < mutation_rate:
            path[i], path[-i] = path[-i], path[i]
    return path

def generate_fitness(path:int, adjacency_matrix):
    """ Calculate path distance using adjacency matrix. """
    distance = 0
    for i in range(len(path)):
        distance += adjacency_matrix[path[i-1]][path[i]]
    return distance


def TSPwGenAlgo(
        g,
        max_num_generations=2000,
        population_size=500,
        mutation_rate=0.02,
        explore_rate=0.9
    ):
    """ A genetic algorithm to attempt to find an optimal solution to TSP. Returns dict with final solution path, final distance, and a list of the avg each gen. """

    # NOTE: YOU SHOULD CHANGE THE DEFAULT PARAMETER VALUES ABOVE TO VALUES YOU
    # THINK WILL YIELD THE BEST SOLUTION FOR A GRAPH OF ~100 VERTS AND THAT CAN
    # RUN IN 5 MINUTES OR LESS (ON AN AVERAGE LAPTOP COMPUTER)

    solution_path = [] # list of n+1 verts representing sequence of vertices with lowest total distance found
    solution_distance = INF # distance of final solution path, note this should include edge back to starting vert
    avg_path_each_generation = [] # store average path length path across individuals in each generation
    path_length = len(g)
    best_ever_solution_length = INF
    best_ever_solution = []

    # create the population 
    population = []
    sum_fitness = 0
    #create individuals in the population in a tuple with their fitness and path
    for _ in range(population_size):
        path = generate_path(path_length)
        fitness = generate_fitness(path, g)
        population.append((fitness, path))
        sum_fitness += fitness
    avg_path_each_generation.append(sum_fitness / population_size)

    # loop for x number of generations (can also choose to add other early-stopping criteria)
    for gen in range(1, max_num_generations):

        #sort the population by the fitness of each individual
        population.sort(key=lambda y: y[0])
        #check if there is a new best ever path and replace if neccessary 
        if population[0][0] < best_ever_solution_length:
            best_ever_solution_length = population[0][0]
            best_ever_solution = population[0][1]
        # select the individuals to be used to spawn the generation, using the exploration rate
        number_explored = int(population_size * explore_rate)
        parents = population[:number_explored]

        population = []
        sum_fitness = 0

        # then create individuals of the new generation (using some form of crossover)
        for _ in range(population_size):
            #randomly sample 2 individuals from the last generation
            couple = sample(parents, 2)
            #crossover the genes
            path = generate_child_path(couple[0][1], couple[1][1])
            # allow for mutations 
            path = mutation(path, mutation_rate)
            # calculate fitness of path
            fitness = generate_fitness(path, g)
            population.append((fitness, path))
            sum_fitness += fitness
        
        #narrow exploration rate as generations progress
        if explore_rate > 0.22:
            explore_rate = explore_rate * .95

        # calculate average path length across individuals in this generation and store in avg_path_each_generation
        avg_path_each_generation.append(sum_fitness / population_size)

    # calculate final solution
    population.sort(key=lambda y: y[0])
    if population[0][0] < best_ever_solution_length:
            best_ever_solution_length = population[0][0]
            best_ever_solution = population[0][1]

    #print details & save graph of averages over time
    print(f"The max gens was {max_num_generations} and the pop size was {population_size}")
    print(f"the best ever solution path was length: {best_ever_solution_length}")
    print(f"the best ever solution path was {best_ever_solution}")
    gens = list(range(max_num_generations))
    plt.plot(gens, avg_path_each_generation)
    plt.xticks(gens)
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.savefig("fitnessovertime.png")

    # update solution_path and solution_distance
    solution_distance, solution_path = population[0][0], population[0][1]

    return {
            'solution_path': solution_path,
            'solution_distance': solution_distance,
            'evolution': avg_path_each_generation
           }


def TSPwDynProg(g):
    """ (10pts extra credit) A dynamic programming approach to solve TSP """
    solution_path = [] # list of n+1 verts representing sequence of vertices with lowest total distance found
    solution_distance = INF # distance of solution path, note this should include edge back to starting vert

    #...

    return {
            'solution_path': solution_path,
            'solution_distance': solution_distance,
           }


def TSPwBandB(g):
    """ (10pts extra credit) A branch and bound approach to solve TSP """
    solution_path = [] # list of n+1 verts representing sequence of vertices with lowest total distance found
    solution_distance = INF # distance of solution path, note this should include edge back to starting vert

    #...

    return {
            'solution_path': solution_path,
            'solution_distance': solution_distance,
           }


def assign05_main():
    """ Load the graph (change the filename when you're ready to test larger ones) """
    g = adjMatFromFile("complete_graph_n100.txt")

    # Run genetic algorithm to find best solution possible
    start_time = time.time()
    res_ga = TSPwGenAlgo(g)
    elapsed_time_ga = time.time() - start_time
    print(f"GenAlgo runtime: {elapsed_time_ga:.2f}")
    print(f"  sol dist: {res_ga['solution_distance']}")
    print(f"  sol path: {res_ga['solution_path']}")

    # (Try to) run Dynamic Programming algorithm only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_dyn_prog = TSPwDynProg(g)
        elapsed_time = time.time() - start_time
        if len(res_dyn_prog['solution_path']) == len(g) + 1:
            print(f"Dyn Prog runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_dyn_prog['solution_distance']}")
            print(f"  sol path: {res_dyn_prog['solution_path']}")

    # (Try to) run Branch and Bound only when n_verts <= 10
    if len(g) <= 10:
        start_time = time.time()
        res_bnb = TSPwBandB(g)
        elapsed_time = time.time() - start_time
        if len(res_bnb['solution_path']) == len(g) + 1:
            print(f"Branch & Bound runtime: {elapsed_time:.2f}")
            print(f"  sol dist: {res_bnb['solution_distance']}")
            print(f"  sol path: {res_bnb['solution_path']}")


# Check if the program is being run directly (i.e. not being imported)
if __name__ == '__main__':
    assign05_main()
