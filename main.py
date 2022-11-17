import random
import math
import numpy as np
import matplotlib.pyplot as plt

p = 200
g = 200
c1 = random.uniform(0, 4)
c2 = 4 - c1


# Taking input from the user
option = int(input("What would you like to find the solution for? "
                   "Type 1 for Eggholder, type 2 for Holder table."))

# Setting the x and y limits
if option == 1:
    lower = -512
    upper = 512
else:
    lower = -10
    upper = 10


# To result the value of the objective function, which gives the fitness of a particle
def objective_function(pi):

    x = pi[0]
    y = pi[1]

    # Egg holder function
    if option == 1:
        return -(y+47)*math.sin(math.sqrt(abs(x/2 + (y+47)))) - x*math.sin(math.sqrt(abs(x-(y+47))))

    # Holder table function
    if option == 2:
        return -abs(math.sin(x)*math.cos(y)*math.exp(abs(1 - (math.sqrt(x**2 + y**2))/math.pi)))


# To define a population
def generate_population():

    # Choosing p random x and y
    x = np.random.uniform(lower, upper, size=p)
    y = np.random.uniform(lower, upper, size=p)

    population = []

    # Forming p vectors using the above random x and y
    for member in range(p):
        population.append([x[member], y[member]])

    return population


def velocity_update(vi, pi, pibest, gbest, rand):

    return [vi[0] + c1*rand*(pibest[0] - pi[0]) + c2*rand*(gbest[0] - pi[0]),
            vi[1] + c1*rand*(pibest[1] - pi[1]) + c2*rand*(gbest[1] - pi[1])]


def position_update(pi, vinew):

    return [(pi[0] + vinew[0]), (pi[1] + vinew[1])]


def limiting_within_constraints(pi):

    # Checking if x and y are in the given range, and replacing with the bound otherwise
    if pi[0] > upper:
        pi[0] = upper
    if pi[0] < lower:
        pi[0] = lower
    if pi[1] > upper:
        pi[1] = upper
    if pi[1] < lower:
        pi[1] = lower


# To plot a set of vectors
def plot(population, symbol, label):

    x = []
    y = []

    # Taking the elements of vectors into x and y lists
    for member in range(len(population)):
        x.append(population[member][0])
        y.append(population[member][1])

    plt.plot(x, y, symbol, label=label)


# Generating the population
particles = generate_population()

# Plotting the initial population
plt.title("Initial particle positions")
plot(particles, "+", "Particle")
plt.show()

# best position in the particle's life
p_best = []
# best fitness attained in the particle's life
f_best = []

# Initializing with zero velocity for each particle
v = []
for i in range(p):
    p_best.append(particles[i])
    f_best.append(objective_function(particles[i]))
    v.append([0, 0])

# global best - the best position any particle has ever attained
g_best = particles[f_best.index(min(f_best))]


# To find average and best fitness
def fitness(population):

    fitness_list = []
    for particle in population:
        fitness_list.append(objective_function(particle))

    return sum(fitness_list)/len(fitness_list), min(fitness_list)


average_fitness, best_fitness = [fitness(particles)[0]], [fitness(particles)[1]]


# Creating the for loop to run across all the generations
for gen in range(g):

    # Initializing the particle index
    i = 0

    # Creating a while loop to run across all the particles in the population
    while i in range(p):

        # Generating a random number for rand in the v_new formula
        r = random.random()

        # Updating the velocity of the particle
        v[i] = velocity_update(v[i], particles[i], p_best[i], g_best, r)

        # Updating the position of the particle
        particles[i] = position_update(particles[i], v[i])

        # Confining the particle's new position within the bounds
        limiting_within_constraints(particles[i])

        # Updating the particle's pbest and fbest
        if objective_function(particles[i]) < objective_function(p_best[i]):
            p_best[i] = particles[i]
            f_best[i] = objective_function(p_best[i])

        # Incrementing the particle index to go the next particle
        i = i+1

    # Updating the gbest
    if min(f_best) < objective_function(g_best):
        g_best = p_best[f_best.index(min(f_best))]
    average_fitness.append(fitness(particles)[0])
    best_fitness.append(fitness(particles)[1])


# PLOTTING:

# Plotting the particles after swarm optimization
plt.title("Particles after Swarm Optimization")
plot(particles, "+", "Particle")
plt.show()

# Plotting the average fitness and best fitness across generations
plt.title("Convergence history")
plt.plot(average_fitness, label="Average fitness")
plt.plot(best_fitness, label="Best fitness")
plt.legend(loc="upper left")
plt.show()

# Outputting the minimum of the function and the gbest position
print(f"Minimum = {min(f_best)} at gbest = {g_best}")
