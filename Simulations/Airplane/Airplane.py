import numpy as np

class Airplane():
    def __init__(
            self, 
            n_rows: int, 
            seat_configurations:tuple, 
            row_distance: float, 
            **kwargs
    ):
        self.n_rows = n_rows
        self.seat_configurations = seat_configurations
        self.total_seats = n_rows *sum(self.seat_configurations)
        self.row_distance = row_distance

    def order_chromosome(self, chromosome):
        argsort_chromosome= np.argsort(chromosome.chromosome)
        #inverse_mapping = np.argsort(argsort_chromosome)
        sorted_chromosome = chromosome.chromosome[argsort_chromosome]

        return sorted_chromosome, argsort_chromosome#, inverse_mapping
 
    def create_passenger_groups(self, chromosome, group_length):
        ordered_chromosome, argsort = self.order_chromosome(chromosome)
        
        index = 0
        groups = []
        while index <= self.total_seats-1:
            group = ordered_chromosome - ordered_chromosome[index]
            for i in range(index, self.total_seats):
                if group[i] > group_length:
                    orginal_index = max(argsort[index:i])
                    index = i
                    groups.append((index-1, orginal_index))
                    break
                elif i == self.total_seats -1:
                    original_index = max(argsort[index:])
                    index = i
                    groups.append((index, original_index))
                    index = self.total_seats
                
        return groups
    
    def compute_group_disembarking_time(self, last_group_passenger_row, passenger_speed):
        """
        computes the disembarking time of the group. This is assuming all passengers walk 
        at the same speed
        """
        time = ((last_group_passenger_row +1)* self.row_distance)/passenger_speed

        return time
    
    def compute_total_disembarking_time(self, groups, passenger_speed):
        total_time = 0
        for group in groups:
            total_time += self.compute_group_disembarking_time(group[1], passenger_speed)
        
        return total_time
    
    def compute_fitness(self, time, min_time):
        """
        This function allows to transform the disembarking time of the plane into a
        number between 0 and 1, setting the greatest value to the shortest time. It 
        assumes that there exist a minimum possible time of disembarking. The 
        function has an exponential behavior exp(-(T-min_T)/min_T).
        """
        return np.exp(-(time/min_time-1))
    
    def compute_min_time(self, mean_speed):
        """
        for the first use case the min time will be the mean speed times the distance 
        of the plane.
        """
        return self.row_distance*self.n_rows*mean_speed

