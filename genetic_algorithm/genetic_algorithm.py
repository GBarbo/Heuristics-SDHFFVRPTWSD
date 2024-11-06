class Individual:
    def __init__(self, routes, partial_demands, vehicles, customers, d, t, R):
        self.routes = routes
        self.partial_demands = partial_demands
        self.feasibility = True
        self.fitness = self.calculate_fitness(routes)

    def calculate_fitness(self, routes):
        # Calculates fitness function and determines feasibility

        cost = 0
        for route, vehicle_id in routes:
            for i in range(1, len(route)):
                cost += d[route[i-1]][route[i]] * vehicles[vehicle_id].freight_cost

        time_infeasibility, site_infeasibility, capacity_infeasibility = 0, 0, 0

        for route, vehicle_id in routes:
            # Time infeasibility
            if not check_time_windows(t, route, customers):
                self.feasibility = False
                time_infeasibility += 1

            # Site infeasibility
            for i in range(1, len(route) - 1):
                if not check_site_dependency(vehicles[vehicle_id], route[i]):
                    self.feasibility = False
                    site_infeasibility += 1
            
            # Capacity infeasibility 
            if not check_vehicle_capacity(route, vehicle_id, self.partial_demands[vehicle_id]):
                self.feasibility = False
                capacity_infeasibility += 1
        
        # Fitness as a function of cost and weighted infeasibilities
        fitness = cost + 10000 * time_infeasibility + 10000 * site_infeasibility + 10000 * capacity_infeasibility
        
        return(fitness)
