## Assign 5

### Coding directions

In this fifth assignment we will be implementing a Genetic Algorithm to solve
the Traveling Salesperson problem for a complete, weighted, and directed graph. 

### Required: Function to be implemented

Note that only one function is required for this assignment (although you may
choose to implement auxiliary/helper functions). The required function must have
the exact name and parameters shown below since your program will be
automatically tested. 

Note that you should select default values for the parameters that you think
will work best for a graph with approximately 100 vertices and that will not
cause your function to run for more than ~5 minutes on an average computer. That
means you will put specific values in for the 'xyz' shown below that you think
will allow your function to return the tour with the lowest cost or distance. 

* __TSPwGenAlgo(W, max_num_generations=xyz, population_size=xyz, mutation_rate=xyz, explore_rate=xyz)__
  * input: 
      an adjacency weight matrix, 
      maximum number of generations, 
      population size to use, 
      mutation rate, and 
      exploration rate 
        (e.g. a smaller exploration rate
        could mean that a smaller group of 'fit' individuals should be used for reproduction,
        while a larger rate could mean a larger group of individuals should be used)
  * output/return: a Python __dictionary__ containing: 
      the solution path (i.e.list of n+1 vertices representing the traveling salesperson solution), 
      the distance of the solution path, 
      and a list with the shortest distance found in each generation __(see assign5.py for more specific details)__ 

### Optional: Each of these functions implemented (correctly) is +10 pts

* __TSPwDynProg(W)__ 
  * input: adjacency weight matrix, W such that W[i][j] is the weight from vertex i to vertex j
  * output/return: a dictionary containing the solution path and distance (see assign5.py) 

* __TSPwBandB(W)__ carry out Dijkstra's algorithm on a graph represented by its weight matrix
  * input: adjacency weight matrix, W such that W[i][j] is the weight from vertex i to vertex j
  * output/return: a dictionary containing the solution path and distance (see assign5.py) 


Since your program will not be run through unit test on GitHub, there will also
not be any linting checks. However, be sure to document your code, including any
auxiliary functions that you use. You should replace/change the existing
comments that are there to help you get started. 

