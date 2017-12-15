#import random
import numpy as np
from deap import creator, base, tools, algorithms
#code originally from 1eesh

BASEORDER = list("ACGT")

def getRandomBase(randomizer,args) :
	return randomizer.choice(BASEORDER , p=args['nucleotide_frequency'] ) 

def fitness(motif, individual):
	return (''.join(individual).count(motif)),

def mutateSequence(randomizer, args, individual):
	for i in xrange(len(individual)):
		if randomizer.random() < args['mutation_frequency']:
			individual[i]=getRandomBase(randomizer, args)
	return individual, 

	
def seqsToOHC(seqX):
	ohcX = np.zeros((np.shape(seqX)[0],4,np.shape(seqX)[1],1))
	for i in range(0,4):#bases
		for j in range(0,np.shape(seqX)[0]):
			ohcX[j,i,[x==BASEORDER[i] for x in seqX[j,:]],:]=1;
	return (ohcX)

def	evaluateThese(sess, seqX, args):
	numSeqs=np.shape(ohcX)[0]
	ohcX = seqsToOHC(seqX);
	predY = np.zeros(numSeqs);
	z=0
	while z < numSeqs:
		curPredY = sess.run([predELY], feed_dict={ohcX: ohcX[z:(z+args['batch_size']),:,:,:]})
		predY[z:(z+args['batch_size'])] = curPredY; 
		z=z+args['batch_size']
	return (predY);

def seqSelectionRound(sess, population, toolbox, args):
	offspring = algorithms.varAnd(population, toolbox, cxpb=args['cspb'], mutpb=args['mutpb'])
	fits = evaluateThese(sess, offspring,args);
	for fit, ind in zip(fits, offspring):
		ind.fitness.values = fit
	population = toolbox.select(offspring, k=len(population))
	return (population);
	



def runExample():
	args  = {'sequence_length' : 110 , 'nucleotide_frequency' : [0.25,0.25,0.25,0.25], 'mutation_frequency' : 0.05 } 
	np.random.seed(1239571243);
	randomizer=np.random
	creator.create("FitnessMax", base.Fitness, weights=(1.0,))
	creator.create("Individual", list , fitness=creator.FitnessMax)

	toolbox = base.Toolbox()
	toolbox.register("base", getRandomBase, randomizer, args)
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.base, n=args['sequence_length'])
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("evaluate", fitness, "ATGC")
	#toolbox.register("evaluate", fitness)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", mutateSequence, randomizer, args)
	#toolbox.register("mutate", mutateSequence, args['mutation_frequency'])
	toolbox.register("select", tools.selTournament, tournsize=3)

	population = toolbox.population(n=300)
	NGEN=1000
	for gen in range(NGEN):
		offspring = algorithms.varAnd(population, toolbox, cxpb=0.1, mutpb=0.1)
		#offspring = algorithms.varAnd(population, toolbox, cxpb=0.1)
		fits = toolbox.map(toolbox.evaluate, offspring)
		print(type(fits))
		for fit, ind in zip(fits, offspring):
			ind.fitness.values = fit
		population = toolbox.select(offspring, k=len(population))
	top10 = tools.selBest(population, k=10)
	return(top10);

