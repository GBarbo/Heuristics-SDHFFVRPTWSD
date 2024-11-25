# ==================================================
# Genetic Algorithm for the SDHFFVRPTWSD
# Author: Giovanni Cesar Meira Barboza
# Version: Starting population with Clarke-Wright randomized
# Date: 2024-11-19
# Description: genetic algorithm with SREX crossover to solve the SDHFFVRPTWSD 
# ==================================================

import random
from clarke_wright_randomized import clarke_wright_randomized
from clarke_wright_randomized import check_time_windows
from clarke_wright_randomized import check_time_windows_concomitance
from clarke_wright_randomized import check_site_dependency
from clarke_wright_randomized import concomitance_detection
from clarke_wright_randomized import concomitance_wait

def deep_copy(obj):
    if isinstance(obj, list):
        return [deep_copy(element) for element in obj]
    else:
        return obj
    
class customer:
    def __init__(self, id, demand, start_time, service_time, end_time):
        self.id = id             # Customer ID
        self.demand = demand     # Demand (ton)
        self.start_time = start_time  # Start time (hours)
        self.service_time = service_time  # Service time (hours)
        self.end_time = end_time      # End time (hours)

class vehicle:
    def __init__(self, id, capacity, freight_cost, R):
        self.id = id
        self.capacity = capacity
        self.freight_cost = freight_cost
        self.R = R

def check_vehicle_capacity(route, vehicle_id, vehicle_partial_demands, customers, vehicles):
    load = 0
    for i in range(1, len(route) - 1):
        load += vehicle_partial_demands[route[i]] * customers[route[i]].demand
        if load > vehicles[vehicle_id].capacity:
            return False
    return True

class Individual:
    def __init__(self, id, routes, partial_demands, wait_times, customers, vehicles, d, t):
        self.id = id
        self.routes = routes
        self.partial_demands = partial_demands
        self.wait_times = wait_times
        self.feasibility = True  # Attribute to store feasibility
        self.fitness = None  # Initialize fitness as None
        
        # Store parameters for use in update_fitness
        self.customers = customers
        self.vehicles = vehicles
        self.d = d
        self.t = t

        self.update_fitness()  # Initial calculation of fitness and feasibility

    def update_fitness(self):
        cost = 0
        for route, vehicle_id in self.routes:
            for i in range(1, len(route)):
                cost += self.d[route[i - 1]][route[i]] * self.vehicles[vehicle_id].freight_cost

        time_infeasibility, site_infeasibility, capacity_infeasibility, concomitance_infeasibility = 0, 0, 0, 0
        self.feasibility = True  # Reset feasibility

        for route, vehicle_id in self.routes:
            # Time infeasibility
            if not check_time_windows_concomitance(
                self.wait_times[self.routes.index([route, vehicle_id])],
                self.t,
                route,
                self.customers
            ):
                self.feasibility = False
                if not check_time_windows(self.t, route, self.customers):
                    time_infeasibility += 1
                else:
                    concomitance_infeasibility += 1

            # Site infeasibility
            for i in range(1, len(route) - 1):
                if not check_site_dependency(self.vehicles[vehicle_id], route[i]):
                    self.feasibility = False
                    site_infeasibility += 1

            # Capacity infeasibility
            if not check_vehicle_capacity(route, vehicle_id, self.partial_demands[vehicle_id], self.customers, self.vehicles):
                self.feasibility = False
                capacity_infeasibility += 1

        # Total wait calculation, ADD TO FITNESS FUNCTION AFTER DEBUGGING
        total_wait = 0
        for wait_route in self.wait_times:
            for wait_customer in wait_route:
                total_wait += wait_customer

        # Fitness as a function of cost, wait times, and weighted infeasibilities
        self.fitness = cost + 10000 * time_infeasibility + 10000 * site_infeasibility + 10000 * capacity_infeasibility + 500 * concomitance_infeasibility

    def set_routes(self, new_routes):
        self.routes = new_routes
        self.update_fitness()  # Recalculate fitness when routes are updated

def swap_intra_route(route, i, j):
    new_route = route[:]
    
    if 1 <= i < len(new_route) - 1 and 1 <= j < len(new_route) - 1:
        new_route[i], new_route[j] = new_route[j], new_route[i]
    
    return new_route

def sus_selection(population):
    # Input: population of individuals
    # Output: population of paired parents of size equal to the individuals selected through SUS

    # Calculate and sort probabilities based of fitness 
    sum_fitness = 0
    for individual in population:
        sum_fitness += 1/individual.fitness

    probabilities = []
    for individual in population:
        probabilities.append([1/individual.fitness / sum_fitness, individual.id])  # The inverse fitness because the higher is this value, the worse is the solution
        
    sorted_probabilities = sorted(probabilities,  key=lambda x: x[0], reverse=True)
    
    # Transform probabilities in cummulative to perform SUS
    current_prob = 0
    for probability in sorted_probabilities:
        prob = probability[0]
        probability[0] = current_prob + probability[0]
        current_prob += prob

    # SUS select the parents
    parents = []
    couple = []

    for _ in range(len(population)):
        pointer = random.random()
        for i in range(len(population)):
            if pointer < sorted_probabilities[i][0]:
                couple.append(sorted_probabilities[i][1])
                break
        else:
            couple.append(sorted_probabilities[len(population) - 1][1])
        
        if len(couple) == 2:
            parents.append(couple)
            couple = []
    
    return parents

def evaluate_route(route, d):
    # Route evaluation based on distance and number of customers ratio

    total_distance = 0
    for i in range(1, len(route)):
        total_distance += d[route[i-1]][route[i]]

    return total_distance / len(route)

def best_worst_routes(routes, d):
    # Input: routes and distance (d) matrices
    # Output: best and worst subroute indices (respectively) according to the ratio (total distance)/(number of customers)

    subroutes = [route[0] for route in routes]
    evaluation = []
    for i in range(len(subroutes)):
        evaluation.append([i, evaluate_route(subroutes[i], d)])
    
    evaluation = sorted(evaluation,  key=lambda x: x[1])
    best_route_id = evaluation[0][0]
    worst_route_id = evaluation[len(evaluation)-1][0]

    return best_route_id, worst_route_id

def led_position(route, customer, d):
    # Find the best position to insert the customer accordign to the least-extra-distance

    led = []
    for i in range(1, len(route)):
        led.append([d[route[i-1]][customer] + d[customer][route[i]] - d[route[i-1]][route[i]], i])

    led = sorted(led,  key=lambda x: x[0])

    return led[0][1]

def customer_service(partial_demands):
    # Fraction of the customers demands currently serviced by the partial_demands vector

    serviced_demands = [0.0] * (len(partial_demands[0]))
    for i in range(len(partial_demands)):
        for j in range(len(partial_demands[0])):
            serviced_demands[j] += partial_demands[i][j]    

    return(serviced_demands)

def srex_crossover(routes1, routes2, partial_demands1, partial_demands2, customers, vehicles, d):
    # Input: routes and partial demands of both parents customers list, vehicles list and distance matrix (d)
    # Output: routes and partial demands for both offsprings generated by SREX

    # Deep copies are necessary to avoid altering the parents
    p1 = deep_copy(routes1)
    p2 = deep_copy(routes2)
    f1 = deep_copy(partial_demands1)
    f2 = deep_copy(partial_demands2)

    best1_id, worst1_id = best_worst_routes(p1, d)
    best2_id, worst2_id = best_worst_routes(p2, d)

    # Offspring 1

    # Remove worst route from p1 and add best route from p2
    o1 = deep_copy(p1)
    fo1 = deep_copy(f1)
    o1.remove(o1[worst1_id])
    fo1.remove(fo1[worst1_id])
    o1.insert(worst1_id, [p2[best2_id][0], p1[worst1_id][1]])
    fo1.insert(worst1_id, f2[best2_id])

    # Identify overservice and underservice
    serviced_demands = customer_service(fo1)
    underserviced = []
    overserviced = []
    for i in range(1, len(serviced_demands)):
        if serviced_demands[i] < 0.9999:
            underserviced.append([i, 1 - serviced_demands[i]])
        elif serviced_demands[i] > 1.0001:
            overserviced.append([i, serviced_demands[i] - 1])

    # Remove overserviced customers from the old routes
    for k in overserviced:
        customer_id, overservice = k
        for i in range(len(o1)):
            if i == worst1_id:  # Do not change the newly added route
                continue
            elif customer_id in o1[i][0]:
                service = min(k[1], fo1[i][customer_id])
                serviced_demands[customer_id] -= service
                fo1[i][customer_id] -= service
                serviced_demands[customer_id]
                k[1] -= service
                if fo1[i][customer_id] < 0.0001:
                    o1[i][0].remove(customer_id)
                if serviced_demands[customer_id] < 1.0001:
                    break

    # Add back underserviced customers
    for k in underserviced:
        # Find route with customer underserviced and fill it with underservice
        customer_id, underservice = k
        for i in range(len(o1)):
            if i == worst1_id:  # Do not change the newly added route
                continue
            if customer_id in o1[i][0]:
                service = min(k[1], 1.0 - fo1[i][customer_id])
                serviced_demands[customer_id] += service
                k[1] -= service
                fo1[i][customer_id] += service
                if serviced_demands[customer_id] > 0.9999:
                    break

        # If the customer is still underserviced, insert it in the route with most available capacity using LED trying
        if serviced_demands[customer_id] < 0.9999:
            available_capacity = []
            for i in range(len(o1)):
                route, vehicle_id = o1[i]
                if customer_id in route:
                    available_capacity.append(-999) # Prevent inserting in route where the customer is already there
                    continue

                load = 0
                for j in range(len(route)):
                    load += fo1[i][j] * customers[route[j]].demand
                
                available_capacity.append(vehicles[vehicle_id].capacity - load)
            
            route_id = available_capacity.index(max(available_capacity))
            o1[route_id][0].insert(led_position(o1[route_id][0], customer_id, d), customer_id)
            fo1[route_id][customer_id] += k[1]
            serviced_demands[customer_id] += k[1]
                
    # Offspring 2

    # Deep copies are necessary to avoid altering the parents
    p1 = deep_copy(routes1)
    p2 = deep_copy(routes2)
    f1 = deep_copy(partial_demands1)
    f2 = deep_copy(partial_demands2)

    # Remove worst route from p2 and add best route from p1

    o2 = deep_copy(p2)
    fo2 = deep_copy(f2)
    o2.remove(o2[worst2_id])
    fo2.remove(fo2[worst2_id])
    o2.insert(worst2_id, [p1[best1_id][0], p2[worst2_id][1]])
    fo2.insert(worst2_id, f1[best1_id])

    # Identify overservice and underservice
    serviced_demands = customer_service(fo2)
    underserviced = []
    overserviced = []
    for i in range(1, len(serviced_demands)):
        if serviced_demands[i] < 0.9999:
            underserviced.append([i, 1 - serviced_demands[i]])
        elif serviced_demands[i] > 1.0001:
            overserviced.append([i, serviced_demands[i] - 1])

    # Remove overserviced customers from the old routes
    for k in overserviced:
        customer_id, overservice = k
        for i in range(len(o2)):
            if i == worst2_id:  # Do not change the newly added route
                continue
            elif customer_id in o2[i][0]:
                service = min(k[1], fo2[i][customer_id])
                serviced_demands[customer_id] -= service
                fo2[i][customer_id] -= service
                serviced_demands[customer_id]
                k[1] -= service
                if fo2[i][customer_id] < 0.0001:
                    o2[i][0].remove(customer_id)
                if serviced_demands[customer_id] < 1.0001:
                    break
    
    # Add back underserviced customers
    for k in underserviced:
        # Find route with customer underserviced and fill it with underservice
        customer_id, underservice = k
        for i in range(len(o2)):
            if i == worst2_id:  # Do not change the newly added route
                continue
            if customer_id in o2[i][0]:
                service = min(k[1], 1.0 - fo2[i][customer_id])
                serviced_demands[customer_id] += service
                k[1] -= service
                fo2[i][customer_id] += service
                if serviced_demands[customer_id] > 0.9999:
                    break

        # If the customer is still underserviced, insert it in the route with most available capacity using LED trying
        if serviced_demands[customer_id] < 0.9999:
            available_capacity = []
            for i in range(len(o2)):
                route, vehicle_id = o2[i]
                if customer_id in route:
                    available_capacity.append(-999) # Prevent inserting in route where the customer is already there
                    continue

                load = 0
                for j in range(len(route)):
                    load += fo2[i][j] * customers[route[j]].demand
                
                available_capacity.append(vehicles[vehicle_id].capacity - load)
            
            route_id = available_capacity.index(max(available_capacity))
            o2[route_id][0].insert(led_position(o2[route_id][0], customer_id, d), customer_id)
            fo2[route_id][customer_id] += k[1]
            serviced_demands[customer_id] += k[1]

    return([o1,fo1],[o2,fo2])

def crossover(population, parents, customers, vehicles, d, t):
    offspring = []  # List of individuals
    idx = 50
    for couple in parents:
        i, j = couple
        p1 = population[i].routes
        f1 = population[i].partial_demands
        p2 = population[j].routes
        f2 = population[j].partial_demands
        
        # Perform crossover between parents
        offspring1, offspring2 = srex_crossover(p1, p2, f1, f2, customers, vehicles, d)

        offspring1_routes, offspring1_partial_demands = offspring1
        offspring2_routes, offspring2_partial_demands = offspring2

        # Detect concomitances and add wait if necessary
        offspring1_concomitances = concomitance_detection(offspring1_routes, customers, t)
        offspring1_wait_times = concomitance_wait(offspring1_concomitances, offspring1_routes, customers, t)
        offspring2_concomitances = concomitance_detection(offspring2_routes, customers, t)
        offspring2_wait_times = concomitance_wait(offspring2_concomitances, offspring2_routes, customers, t)

        # Append new individuals to offspring
        offspring.append(Individual(idx, offspring1_routes, offspring1_partial_demands, offspring1_wait_times, customers, vehicles, d, t))
        idx += 1
        offspring.append(Individual(idx, offspring2_routes, offspring2_partial_demands, offspring2_wait_times, customers, vehicles, d, t))
        idx += 1

    return offspring

def crrm_mutation(population, mutation_rate):
    for individual in population:
        pointer = random.random()
        if pointer > mutation_rate:
            continue
        
        # Randomly choose individual and route to perform swap
        routes = deep_copy(individual.routes)  # Use deep copy if necessary to avoid reference issues
        route_id = random.randint(0, len(routes) - 1)
        route = routes[route_id][0][:]  # Copy the route to mutate
        if len(route) < 4:
            continue

        i = random.randint(1, max(len(route) - 3, 1))
        
        # Perform intra-route swap
        route = swap_intra_route(route, i, i + 1)
        routes[route_id][0] = route
        
        # Update routes and fitness
        individual.routes = routes
        individual.update_fitness()
        
        # Recalculate wait times based on the new routes
        concomitances = concomitance_detection(routes, individual.customers, individual.t)
        wait_times = concomitance_wait(concomitances, routes, individual.customers, individual.t)
        individual.wait_times = wait_times

    return population

def survival(population, survivors_number):
    # Input: population to be subject to survival and number of survivors
    # Output: population of survivors selected by SUS according to the inverse of the fitness

    # Calculate and sort probabilities based on finess
    sum_fitness = 0
    for individual in population:
        sum_fitness += 1/individual.fitness
    probabilities = []
    for individual in population:
        probabilities.append([1/individual.fitness / sum_fitness, population.index(individual)])  # The inverse fitness because the higher is this value, the worse is the solution

    sorted_probabilities = sorted(probabilities,  key=lambda x: x[0], reverse=True)

    # Transform probabilities in cummulative to perform SUS
    current_prob = 0
    for probability in sorted_probabilities:
        prob = probability[0]
        probability[0] = current_prob + probability[0]
        current_prob += prob

    # SUS select the survivors
    survivors = []
    for _ in range(survivors_number):
        pointer = random.random()
        for i in range(len(population)):
            if pointer < sorted_probabilities[i][0]:
                survivors.append(population[sorted_probabilities[i][1]])
                break
        else:
            survivors.append(population[sorted_probabilities[len(population) - 1][1]])

    return survivors

def genetic_algorithm(population_size, mutation_rate, elite_rate, max_iter, customers, vehicles, d, t, R):
    # Input: GA parameters (population_size, mutation_rate) HFVRPTWSP data for customers and vehicles
    # Output: Feasible routes for each vehicle and matrix of split deliveries f [vehicles x customers]

    # Generate initial population
    population = clarke_wright_randomized(customers, vehicles, d, t, R, population_size)

    # Main loop
    counter = 0
    iter_no_improv = 0
    while True:
        # Create population with offspring
        parents = sus_selection(population)
        offspring = crossover(population, parents, customers, vehicles, d, t)
        offspring = crrm_mutation(offspring, mutation_rate)
        population = population + offspring

        # Update fitness for all individuals
        for individual in population:
            individual.update_fitness()

        # Sort population by fitness
        population = sorted(population, key=lambda individual: individual.fitness)

        # Cut population in half by survival, spare elite individuals
        split_index = int(len(population) * elite_rate)
        elite_population = population[:split_index]
        
        best_individual = elite_population[0]
        best_individual.update_fitness()
        best_fitness = best_individual.fitness

        if counter == 0:
            incumbent_best_fitness = best_fitness
        else:
            if best_fitness < incumbent_best_fitness - 0.000001:
                iter_no_improv = 0
                incumbent_best_fitness = best_fitness
            else:
                iter_no_improv += 1

        survivors_number = len(population)//2 - len(elite_population)
        population = population[split_index:]
        survivors = survival(population, survivors_number)
        population = elite_population + survivors
        
        # Restart id
        idx = 0
        for individual in population:
            individual.id = idx
            idx += 1

        counter += 1
        if iter_no_improv > max_iter:
            print(f'Finished after {counter} generations')
            break
    
    return best_individual

def calculate_cost(routes, d, vehicles):
    # Input: list of routes and vehicles assigned, distances matrix and list of vehicles (objects)
    # Output: objective function cost for the mixed fleet VRP

    cost = 0
    for k in routes:
        route = k[0]
        vehicle_id = k[1]
        for i in range(1, len(route)):
            cost += vehicles[vehicle_id].freight_cost * d[route[i-1]][route[i]]
    
    return cost

def main():
    # Parameters
    # Optimum: v0 = 0-4-5-1(0.2)-0, v1 = 0-3-1(0.8)-2-0 

    # Number of clients and vehicles
    n = 5  # number of clients
    V = 2  # number of vehicles

    # Demand (ton)
    q = [0, 18, 0.8, 0.8, 6, 4]

    # Time windows (start, service, and end times)
    e = [8, 12, 10, 8, 8, 8]        # Start time (hours)
    s = [0, 2.5, 1, 1.5, 1.5, 1]    # Service time (hours)
    l = [18, 20, 20, 14, 12, 13]    # End time (hours)

    # Distance matrix (km)
    d = [
        [0, 106, 116, 47, 57, 58],
        [106, 0, 21, 117, 127, 127],
        [116, 21, 0, 119, 128, 128],
        [47, 117, 119, 0, 11, 12],
        [57, 127, 128, 11, 0, 2],
        [58, 127, 128, 12, 2, 0]
    ]

    # Travel time matrix (hours)
    t = [
        [0.00, 2.64, 2.91, 1.17, 1.41, 1.44],
        [2.64, 0.00, 0.54, 2.92, 3.16, 3.16],
        [2.91, 0.54, 0.00, 2.98, 3.20, 3.19],   
        [1.17, 2.92, 2.98, 0.00, 0.28, 0.30],
        [1.41, 3.16, 3.20, 0.28, 0.00, 0.04],
        [1.44, 3.16, 3.19, 0.30, 0.04, 0.00]
    ]

    # Vehicle capacities (ton)
    a = [14, 16]

    # Freight costs (R$/km)
    cf = [4.54, 3.13]

    # Matrix indicating whether vehicle v can deliver to client p (1 for yes, 0 for no)
    R = [
        [1, 0, 0, 1, 1],
        [1, 1, 1, 1, 0]
    ]

    # Create a list of customers using the input data
    customers = [
        customer(id, q[id], e[id], s[id], l[id]) for id in range(n+1)
    ]

    # Create a list of vehicles using the input data
    vehicles = [
        vehicle(id, a[id], cf[id], R[id]) for id in range(V)
    ]

    # Define the Genetic Algorithm parameters
    population_size = 50
    mutation_rate = 0.2
    elite_rate = 0.1
    max_iter = 50   # Maximum number of iterations without improvement to stop

    solution = genetic_algorithm(population_size, mutation_rate, elite_rate, max_iter, customers, vehicles, d, t, R)
    final_routes = solution.routes
    f = solution.partial_demands
    wait = solution.wait_times

    for i in range(len(final_routes)):
        print(f'Vehicle {final_routes[i][1]} route = {final_routes[i][0]}')
    
    for i in range(len(f)):
        print(f'f[{i}] = {f[i]}') 

    for i in range(len(wait)):
        print(f'wait[{i}] = {wait[i]}')

    print(f'Cost = {calculate_cost(final_routes, d, vehicles)}')

if __name__ == "__main__":
    main()