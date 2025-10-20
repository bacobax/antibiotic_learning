# ===========================================
# Antibiotic Resistance Evolution Simulation
# Genome = [membrane, efflux, enzyme, repair]
# ===========================================

import random
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools


# ---------------------------
# 1. Problem and Parameters
# ---------------------------

N_GEN = 200         # Number of generations
POP_SIZE = 100      # Population size
CX_PB = 0.5         # Crossover probability
MUT_PB = 0.3        # Mutation probability
ETA = 0.1           # Mutation strength
SEED = 42
random.seed(SEED)

# Antibiotic environment parameters
kd = 2.0            # antibiotic toxicity
kg = 1.0            # base growth rate
Cr = 0.3            # cost of resistance
A_base = 0.0        # starting antibiotic concentration

# Antibiotic pulse schedule
def antibiotic_concentration(gen):
    """Define antibiotic level over generations"""
    if gen < 10:
        return 0.0
    elif gen < 50:
        return 0.2
    elif gen < 100:
        return 0.4
    else:
        return 0.8

# ---------------------------
# 2. Fitness Function
# ---------------------------

def evaluate(individual, A=None):
    membrane, efflux, enzyme, repair = individual
    if A is None:
        A = A_base

    # Effective antibiotic concentration
    A_eff = A * (1 - 0.2 * efflux) * (1 - 0.6 * enzyme) * (1 - 0.2 * membrane)

    # Survival and growth
    survival = np.exp(-kd * A_eff * (1 - 0.5 * repair))
    growth = kg * (1 - Cr * (membrane + efflux + enzyme + 0.5 * repair))
    growth = max(growth, 0)  # avoid negatives

    fitness = survival * growth
    return (fitness,)

# ---------------------------
# 3. DEAP Setup
# ---------------------------

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.random)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=4)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=ETA, indpb=0.3)
toolbox.register("select", tools.selTournament, tournsize=3)

# ---------------------------
# 4. Evolutionary Loop
# ---------------------------

pop = toolbox.population(n=POP_SIZE)
stats = {"mean_fit": [], "mean_genes": []}

for gen in range(N_GEN):
    A = antibiotic_concentration(gen)

    # Evaluate fitness
    for ind in pop:
        ind.fitness.values = evaluate(ind, A)

    # Record statistics
    fits = [ind.fitness.values[0] for ind in pop]
    mean_fit = np.mean(fits)
    mean_genes = np.mean([ind for ind in pop], axis=0)
    stats["mean_fit"].append(mean_fit)
    stats["mean_genes"].append(mean_genes)

    # Selection, crossover, mutation
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CX_PB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUT_PB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Clamp gene values to [0, 1]
    for ind in offspring:
        for i in range(len(ind)):
            ind[i] = min(max(ind[i], 0), 1)

    pop[:] = offspring

# ---------------------------
# 5. Visualization
# ---------------------------

mean_genes = np.array(stats["mean_genes"])
generations = np.arange(N_GEN)

plt.figure(figsize=(10,5))
plt.plot(generations, stats["mean_fit"], label="Mean Fitness", color="black")
plt.xlabel("Generation")
plt.ylabel("Fitness")
plt.title("Evolution of Mean Fitness")
plt.legend()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(generations, mean_genes[:,0], label="Membrane")
plt.plot(generations, mean_genes[:,1], label="Efflux")
plt.plot(generations, mean_genes[:,2], label="Enzyme")
plt.plot(generations, mean_genes[:,3], label="Repair")
plt.xlabel("Generation")
plt.ylabel("Mean Gene Value")
plt.title("Evolution of Resistance Mechanisms")
plt.legend()
plt.show()
