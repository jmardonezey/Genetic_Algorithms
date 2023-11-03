import numpy as np

class Evolution():
    def __init__(
            self, 
            random_seed, 
            max_generations, 
            n_individuals, 
            n_parents, 
            chromosome_length, 
            chromosome_type, 
            mutation_probability,
            **kwargs
    ):
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.chromosome_length = chromosome_length
        self.chromosome_type = chromosome_type
        self.mutation_probability = mutation_probability
        self.max_generations = max_generations
        self.n_individuals = n_individuals
        self.n_parents = n_parents
        self.crossovers = {
            "one_point": self._one_point_crossover,
            "multiple_points": self._multiple_point_crossover,
            "uniform": self._uniform_crossover
        }
        self.selections = {
            "roulette": self._roulette_selection,
            "ranking": self._ranking_selection,
            "stationary_state": self._stationary_state_selection
        }
        self.evolution = {0:[]}
        self.generation = 0

    
    def populate_1st_generation(self):
        for i in range(self.n_individuals):
            chromosome = Chromosome(self.chromosome_length, self.chromosome_type)
            chromosome.initialize_1st_gen()
            self.evolution[0].append(chromosome)
    
    def evolve(self, commands:dict):
        """
        This method receives a dictionary with the inner commands to operate
        the evolution process. It takes the indiviuals from a generation, 
        select the parents and creates the next generation.
        """
        selection_method = self.selections[commands["selection_method"]]
        crossover_method = commands["crossover_method"]
        selection_inputs = commands["selection_method_inputs"]
        crossover_inputs = commands["crossover_inputs"]
        mutation_inputs = commands["mutation_inputs"]
        stationary_state = bool(commands["stationary_state"])
        remove = commands["remove"]
        individuals = self._create_matriz()
        selected_parents_indices = selection_method(individuals, **selection_inputs)
        parents_indices = selected_parents_indices[:,0].astype("int")

        selected_parents = [self.evolution[self.generation][i] for i in parents_indices]

        self.generation += 1

        if stationary_state:
            stationary_elements_indices = self._stationary_state_selection(individuals, remove)
            stationary_indices = stationary_elements_indices[:,0].astype("int")
            self.evolution[self.generation] = [self.evolution[self.generation-1][i] for i in stationary_indices]
        else:
            self.evolution[self.generation] = []
            remove = 0


        for i in range(self.n_individuals - len(self.evolution[self.generation])):
            parents =tuple(np.random.choice(selected_parents, 2, replace = False))
            new_chromosome = self.apply_crossover(parents, crossover_method, crossover_inputs,**mutation_inputs)
            self.evolution[self.generation].append(new_chromosome)


    def _create_matriz(self):
        """
        this is an inner method, is used to create a matriz that containing the
        relative positions of the chromosomes as saved in the evolution and the
        fitness of each chromosome
        """
        individuals = np.zeros((self.n_individuals, 2))
        for i in range(self.n_individuals):
            individuals[i][0] = i 
            individuals[i][1] = self.evolution[self.generation][i].fitness
        
        return individuals


    def apply_crossover(self, parents:tuple, crossover, crossover_inputs,**mutation_inputs):
        """
        This function receives two chromosomes and returns an new chromosome from the crossover of the parents
        """
        crossover_method = self.crossovers[crossover] 
        if crossover == "one_point":
            order = self._define_parents_order()
            new_genome = crossover_method(parents, order)
        elif crossover == "multiple_points":
            n_points = 2
            new_genome = crossover_method(parents, **crossover_inputs)
        elif crossover == "uniform":
            new_genome = crossover_method(parents)
        new_genome = Chromosome(self.chromosome_length, self.chromosome_type, new_genome)
        #self.mutate(new_genome, self.mutation_probability, **mutation_inputs)
        self.mutate(new_genome, **mutation_inputs)

        return new_genome
    
    def _define_parents_order(self):
        leading = np.random.randint(0,2)
        if leading == 0:
            order = (0,1)
        else:
            order = (1,0)
        return order


    def _one_point_crossover(self, parents:tuple, order:tuple, **kwargs):
        """
        This function manages the method to apply the one point crossover over the parents
        """
        p1, p2 = order
        crossover_point = np.random.randint(0, self.chromosome_length)
        new_genome = np.zeros(self.chromosome_length)
        new_genome[:crossover_point] = parents[p1].chromosome[:crossover_point].copy()
        new_genome[crossover_point:] = parents[p2].chromosome[crossover_point:].copy()

        return new_genome

    def _multiple_point_crossover(self, parents:tuple, n_points:int, **kwargs):
        """
        This function manages the method to apply the multiple points crossover over the parents
        """
        crossover_points = np.random.choice(self.chromosome_length, size = n_points, replace = False)
        crossover_points = np.sort(crossover_points)
        new_genome = np.zeros(self.chromosome_length)
        beginning = 0
        for point in crossover_points:
            p = np.random.randint(0,2)
            if p == 0:
                new_genome[beginning:point] = parents[0].chromosome[beginning:point].copy()
            else:
                new_genome[beginning:point] = parents[1].chromosome[beginning:point].copy()
            beginning = point
        if beginning != (self.chromosome_length -1):
            p = np.random.randint(0,2)
            if p == 0:
                new_genome[beginning:] = parents[0].chromosome[beginning:].copy()
            else:
                new_genome[beginning:] = parents[1].chromosome[beginning:].copy()
        return new_genome


    def _uniform_crossover(self, parents: tuple):
        """
        This function applies the multiple crossover to create the new chromosome, using uniform distribution 
        to select genes from the parents
        """
        new_genome = np.zeros(self.chromosome_length)
        for i in range(self.chromosome_length):
            p = np.random.randint(0,2)
            if p == 0:
                new_genome[i] = parents[0].chromosome[i].copy()
            else:
                new_genome[i] = parents[1].chromosome[i].copy()
        return new_genome

    def mutate(self, genome, **kwargs):
        """
        This method allows to introduce mutations into the new created genome.
        """
        if self.chromosome_type == "binary":
            self.binary_mutation(genome, **kwargs)
        elif self.chromosome_type == "continous":
            self.continous_mutation(genome, **kwargs)
        
    
    def continous_mutation(self, genome, mutation_probability = 0.01, step = 0.3, **kwargs):
        """
        This method allows to introduce continous mutations into the new created genome.
        """
        for i in range(genome.chromosome_length):
            prob = np.random.rand()
            if prob <= mutation_probability:
                step = (np.random.rand()-0.5)*step*2
                genome.chromosome[i] += step
                if genome.chromosome[i] > 1:
                    genome.chromosome[i] = 1
                elif genome.chromosome[i] < 0:
                    genome.chromosome[i] = 0
            else:
                continue

    def binary_mutation(self, genome, mutation_probability = 0.01, **kwargs):
        """
        This method allows to introduce binaries mutations into the new created genome.
        """
        for i in range(genome.chromosome_length):
            prob = np.random.rand()
            if prob <= mutation_probability:
                genome.chromosome[i] = (genome.chromosome[i] + 1)%2
            else:
                continue

    def _parent_selection(self):
        pass

    def _roulette_selection(self, individuals, **kwargs):
        """
        This method receives an array containing the relative positions of the parents
        and their fitness. And then select n_parents based in the roulette method.
        """
        generation = individuals
        fitness = generation[:,1]
        probability = fitness/sum(fitness)
        positions = list(range(self.n_individuals))
        parents_positions = np.random.choice(positions, size = self.n_parents, replace = False, p = probability)
        parents = generation[parents_positions]

        return parents

    def _ranking_selection(self, individuals, sp:float, **kwargs):
        """
        This method receives an array containing the relative positions of the parents
        and their fitness. And then select n_parents based in the ranking method.
        """
        generation = individuals
        argsort = np.argsort(generation[:,1])[::-1]
        probability = np.zeros(self.n_individuals)
        n = self.n_individuals
        for j in range(n):
            i = j+1
            k = argsort[j]
            probability[k] = 1/n*(sp-(2*sp-2)*((i-1)/(n-1)))

        positions = list(range(self.n_individuals))
        parents_positions = np.random.choice(positions, size = self.n_parents, replace = False, p = probability)
        parents = generation[parents_positions]

        return parents

    def _stationary_state_selection(self, individuals, remove):
        """
        This method receives an array containing the relative positions of the parents
        and their fitness. And then select n_parents based in the stationary state method.
        The remove prameter is a float in between 0 and 1, denoting the proportion to remove.
        """
        upper_limit = int(self.n_individuals * (1 -remove))
        generation = individuals
        argsort = np.argsort(generation[:,1])[::-1]
        sorted_parents = generation[argsort]
        parents = sorted_parents[:upper_limit]

        return parents

    def _tournament_selection(self):
        pass

    def _diversity_fitness_selection(self):
        pass


class Chromosome():
    def __init__(self, chromosome_length, chromosome_type, chromosome = None) -> None:
        self.chromosome = chromosome
        self.chromosome_length = chromosome_length
        self.fitness = 0
        self.chromosome_type = chromosome_type
    
    def initialize_1st_gen(self):
        """
        When this function is called a chromosome is initialized
        """
        self.chromosome = self._create_chromosome()

    def _create_chromosome(self):
        """
        This function creates the initial chromosome, depending on the type
        designated.
        """
        if self.chromosome_type == "binary":
            chromosome = np.random.randint(2, size = self.chromosome_length)
        if self.chromosome_type == "continous":
            chromosome = np.random.rand(self.chromosome_length)
        return chromosome 

    




if __name__ == "__main__":
    pass