import random
from deap import creator, base, tools, algorithms
import numpy as np
args  = {'sequence_length' : 5 , 'nucleotide_frequency' :[0.25,0.25,0.25,0.25] } 
randomizer=np.random

def random_sequence_generator(randomizer,args) :
	return randomizer.choice(list('ACGT') , p=args['nucleotide_frequency'] ) 


def fitness(individual):
	return (individual.count('A')),


def mutation(individual, indpb):
	for i in xrange(len(individual)):
		if random.random() < indpb:
			if individual[i]=='A' :
					individual[i] = (randomizer.choice(list('CGT') , p=[args['nucleotide_frequency'][1]/(1-args['nucleotide_frequency'][0]) ,args['nucleotide_frequency'][2]/(1-args['nucleotide_frequency'][0]) ,args['nucleotide_frequency'][3]/(1-args['nucleotide_frequency'][0]) ] ) )
			elif individual[i]=='C' :
					individual[i] = (randomizer.choice(list('AGT') , p=[args['nucleotide_frequency'][0]/(1-args['nucleotide_frequency'][1]) ,args['nucleotide_frequency'][2]/(1-args['nucleotide_frequency'][1]) ,args['nucleotide_frequency'][3]/(1-args['nucleotide_frequency'][1]) ] ) )
			elif individual[i]=='G' :
					individual[i] = (randomizer.choice(list('CGT') , p=[args['nucleotide_frequency'][2]/(1-args['nucleotide_frequency'][2]) ,args['nucleotide_frequency'][1]/(1-args['nucleotide_frequency'][2]) ,args['nucleotide_frequency'][3]/(1-args['nucleotide_frequency'][2]) ] ) )
			elif individual[i]=='T' :
					individual[i] = (randomizer.choice(list('CGT') , p=[args['nucleotide_frequency'][0]/(1-args['nucleotide_frequency'][3]) ,args['nucleotide_frequency'][1]/(1-args['nucleotide_frequency'][3]) ,args['nucleotide_frequency'][2]/(1-args['nucleotide_frequency'][3]) ] ) )
	print(individual)
	print(individual.fitness)
	return individual,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list , fitness=creator.FitnessMax)

toolbox = base.Toolbox()

toolbox.register("base", random_sequence_generator , randomizer , args)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.base, n=args['sequence_length'])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutation, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

population = toolbox.population(n=300)

NGEN=40

for qgen in range(NGEN):
	offspring = algorithms.varAnd(population, toolbox, cxpb=0.1, mutpb=0.1)
	fits = toolbox.map(toolbox.evaluate, offspring)
	for fit, ind in zip(fits, offspring):
		ind.fitness.values = fit
	population = toolbox.select(offspring, k=len(population))

top10 = tools.selBest(population, k=10)

print top10
