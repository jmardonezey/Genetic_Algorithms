import Simulations.Airplane.Airplane as airplane
import  GeneticAlgorithm.tools as ga

conditions = {
    "n_rows": 10, 
    "seat_configurations":(1,), 
    "row_distance": 1,
    "random_seed":0, 
    "max_generations":10, 
    "n_individuals":30, 
    "n_parents":10, 
    "chromosome_length":10, 
    "chromosome_type":"continous", 
    "mutation_probability":0.1
}

commands = {
    "selection_method": "roulette",
    "crossover_method": "one_point",
    "selection_method_inputs": {
        "input1": "input1",
        "input2": "input2"
    },
    "mutation_inputs": {
        "input1": "input1",
        "input2": "input2"
    }
}

class Simulation():
    def __init__(self, n_generations, n_parents, walking_speed):
        self.n_generations = n_generations
        self.n_parents = n_parents
        self.walking_speed = walking_speed

    def initialize_simulation(self, kwargs):
        self.airplane = airplane.Airplane(**kwargs)
        self.evolution_process= ga.Evolution(**kwargs)
        self.evolution_process.populate_1st_generation()
    
    def run_simulation(self, group_length, passenger_speed, commands):
        generation = self.evolution_process.generation
        while generation < self.n_generations:
            print(f"generation {generation}")
            for chromosome in self.evolution_process.evolution[generation]:
                self.get_fitness(chromosome, group_length, passenger_speed)
            self.evolution_process.evolve(commands)
            generation = self.evolution_process.generation
        pass

    def compute_fitness_from_time(self, time):
        """
        This allows to transform the time of disembarking of the simulation
        to the fitness of the chromosome
        """
        
        minimum_time = self.airplane.compute_min_time(self.walking_speed)
        if time - minimum_time < 0:
            return "Error in time"
        else:
            fitness = self.airplane.compute_fitness(time, minimum_time)
            
        return fitness

    def get_fitness(self, chromosome, group_length, passenger_speed):
        groups = self.airplane.create_passenger_groups(chromosome, group_length)
        time = self.airplane.compute_total_disembarking_time(groups, passenger_speed)
        chromosome.fitness = self.compute_fitness_from_time(time)
        
        

    def create_passenger_groups(self):
        pass

if __name__ == "__main__":
    simulation = Simulation(2,4,1)
    simulation.initialize_simulation(conditions)
    simulation.run_simulation(0.2,1,commands)