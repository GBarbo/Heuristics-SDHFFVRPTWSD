# ==================================================
# Clarke-Wright Algorithm for the SDHFFVRPTWSD
# Author: Giovanni Cesar Meira Barboza
# Version: Paralell with starting criterium and concomitance correction
# Date: 2024-11-18
# Description: constructive heuristic to tackle the SDHFFVRPTWSD problem
# ==================================================

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

def calculate_savings(d):
    # Input: triangular matrix of distances (d) between all customers and depot
    # Output: list of customer pairs ranked by savings

    n = len(d) - 1

    # Calculate savings for each pair of clients (i, j)
    savings = []
    for i in range(1, n+1):
        for j in range(i+1, n+1):
            saving_value = d[0][i] + d[0][j] - d[i][j]  # Clarke-Wright's savings formula
            savings.append([i, j, saving_value])

    # Sort savings in descending order
    sorted_savings = sorted(savings, key=lambda x: x[2], reverse=True)
    sorted_savings = [sublist[:-1] for sublist in sorted_savings]   # Removing the saving term

    return sorted_savings

def check_time_windows(t, possible_route, customers):
    # Input: matrix of travel times (t), pssible route and list of customers
    # Output: whether any of the customers or depot in the route disobeys its time window

    time = customers[0].start_time
    for i in range(1, len(possible_route) - 1):
        time += t[possible_route[i-1]][possible_route[i]]
        if time < customers[possible_route[i]].start_time:
            #print(f'start time violation at {possible_route[i]}')
            return False
        time += customers[possible_route[i]].service_time
        if time > customers[possible_route[i]].end_time:
            #print(f'end time violation at {possible_route[i]}')
            return False
        if i == 1 and time - t[possible_route[0]][possible_route[i]] > customers[possible_route[0]].end_time:
            #print(f'depot end time violation')
            return False
        
    return True

def check_site_dependency(vehicle, customer_id):
    # Input: vehicle and customer id
    # Output: whether the vehicle can visit the customer

    if vehicle.R[customer_id - 1] == 0:
        return False
    else:
        return True
    
def merge_route(current_route, pair):
    # Input: route containig either i or j as the first or last customer, pair of customers (i, j)
    # Output: merged route from original route and [0,i,0] or [0,j,0]

    i, j = pair

    if i in current_route:
        if current_route.index(i) == len(current_route) - 2:
            new_route = current_route[:len(current_route) - 1] + [j] + current_route[len(current_route) - 1:]
        elif current_route.index(i) == 1:
            new_route = current_route[:1] + [j] + current_route[1:]
        else:
            raise Exception("i not in route")
    elif j in current_route:
        if current_route.index(j) == len(current_route) - 2:
            new_route = current_route[:len(current_route) - 1] + [i] + current_route[len(current_route) - 1:]
        elif current_route.index(j) == 1:
            new_route = current_route[:1] + [i] + current_route[1:]
        else:
            raise Exception("j not in route")
    else:
        raise Exception("Both not in route")

    return new_route

def check_routes(pair, routes, fully_serviced):
    # Input: pair of customers, list of routes an list of fully serviced customers
    # Output: index of the first route with a merger that could be made, else return negative number

    i, j = pair
    route_id = -2 # If the loop runs and neither i nor j is found, -2 is returned to signal new route

    if i not in fully_serviced and j in fully_serviced:
        for k in range(len(routes)):
            if (i in routes[k][0]) and (j in routes[k][0]):
                continue
            if (j in routes[k][0]):
                route_id = k
                break   # Takes the first vehicle found

    elif j not in fully_serviced and i in fully_serviced:
        for k in range(len(routes)):
            if (i in routes[k][0]) and (j in routes[k][0]):
                continue
            if (i in routes[k][0]):
                route_id = k
                break   # Takes the first vehicle found

    elif i not in fully_serviced and j not in fully_serviced:
        for k in range(len(routes)):
            if (i in routes[k][0]) and (j in routes[k][0]):
                continue
            if (i in routes[k][0]) or (j in routes[k][0]):
                route_id = k
                break   # Takes the first vehicle found
    else:
        raise Exception(f"Both {i} and {j} fully serviced, remove saving pair")
   
    return route_id

def concomitance_detection(routes, customers, t):
    # Input: routes, list of customers and time matrix
    # Output: routes respecting the non concomitance of two or more vehicles in a customer

    # Identify customer and routes where there is concomitance
    concomitance = []   # List of [customer_id, route_id1, route_id2] where there is concomitance

    start_times = []    # List of customers start time from each route
    for i in range(len(routes)):
        time = customers[0].start_time
        route = routes[i][0]
        for j in range(1, len(route) - 1):
            time += t[route[j-1]][route[j]]
            start_times.append([route[j], i, time])   # Each start_time element is stored as [customer_id, route_id, time of arrival]
            time += customers[route[j]].service_time
    
    sorted_start_times = sorted(start_times, key=lambda x: x[2])

    current_customers = []
    for x in sorted_start_times:
        current_customer_id, current_route_id, time = x

        # Check for concomitaces
        for customer_id, route_id, time_of_arrival in current_customers:
            if current_customer_id == customer_id:
                if time < time_of_arrival + customers[customer_id].service_time:
                    concomitance.append([customer_id, route_id, current_route_id])

            # Remove serviced customers from time sweep
            if time >= customer_id + customers[customer_id].service_time:
                current_customers.remove([customer_id, route_id, time_of_arrival])

        # Add customer to current service
        current_customers.append([current_customer_id, current_route_id, time])

    return concomitance

def route_time(customer_id, route, wait, customers, t):
    # Input: customer id, route and wait lists
    # Output: start time of the service in the customer

    time = customers[0].start_time

    for i in range(1, route.index(customer_id) + 1):
        time += customers[route[i-1]].service_time
        time += t[route[i-1]][route[i]]
        if wait[route[i]] > 0.0001:
            time += wait[route[i]]

    return time

def concomitance_wait(concomitances, routes, customers, t):
    # Input: list of concomitances (elements [customer_id, route_id1, route_id2]), routes, list of customers and time matrix
    # Output: wait vector, where each element is the wait the vehicle (row) has to wait to start the service at the customer (line)

    n = len(customers) - 1
    V = len(routes)

    wait = [[0.0] * (n + 1) for _ in range(V)]
    
    for customer_id, route_id1, route_id2 in concomitances:
        time = route_time(customer_id, routes[route_id1][0], wait[route_id1], customers, t)
        time += customers[customer_id].service_time
        wait[route_id2][customer_id] += time - route_time(customer_id, routes[route_id2][0], wait[route_id2], customers, t)
    
    return wait

def check_time_windows_concomitance(wait, t, possible_route, customers):
    # Input: wait times for the route, matrix of travel times (t), pssible route and list of customers
    # Output: whether any of the customers or depot in the route disobeys its time window considering the wait

    time = customers[0].start_time
    for i in range(1, len(possible_route) - 1):
        time += t[possible_route[i-1]][possible_route[i]]
        if wait[i] > 0.0001:
            time += wait[i]
        if time < customers[possible_route[i]].start_time:
            #print(f'start time violation at {possible_route[i]}')
            return False
        time += customers[possible_route[i]].service_time
        if time > customers[possible_route[i]].end_time:
            #print(f'end time violation at {possible_route[i]}')
            return False
        if i == 1 and time - t[possible_route[0]][possible_route[i]] > customers[possible_route[0]].end_time:
            #print(f'depot end time violation')
            return False
        
    return True

def deep_copy(obj):
    if isinstance(obj, list):
        return [deep_copy(element) for element in obj]
    else:
        return obj
    
def swap_intra_route(route, i, j):
    new_route = route[:]
    
    if 1 <= i < len(new_route) - 1 and 1 <= j < len(new_route) - 1:
        new_route[i], new_route[j] = new_route[j], new_route[i]
    
    return new_route

def concomitance_correction(wait, concomitances, routes, customers, t):
    # Try to fix time windows infeasibility by swapping customer with the previous one

    new_wait = deep_copy(wait)
    new_routes = deep_copy(routes)

    for customer_id, route_id1, route_id2 in concomitances:
        if not check_time_windows_concomitance(new_wait[route_id2], t, new_routes[route_id2][0], customers):
            new_wait[route_id2] = [0.0] * (len(customers))

            # Perform swap, avoid with depot
            if customer_id != new_routes[route_id2][0][1]: 
                new_routes[route_id2][0] = swap_intra_route(new_routes[route_id2][0], new_routes[route_id2][0].index(customer_id), new_routes[route_id2][0].index(customer_id) - 1)

            concomitances = concomitance_detection(new_routes, customers, t)
            new_wait = concomitance_wait(concomitances, new_routes, customers, t)
            
            # If new route is time infeasible, avoid swap
            if not check_time_windows_concomitance(new_wait[route_id2], t, new_routes[route_id2][0], customers):
                new_routes = deep_copy(routes)
                new_wait = deep_copy(wait)
                continue

    return new_routes, new_wait

def clarke_wright(customers, vehicles, d, t, R):
    # Input: list of customers, list of vehicles, matrix of distances (d) and matrix of travel times (t) and site dependency matrix (R)
    # Output: Feasible routes for each vehicle, matrix of split deliveries f [vehicles x customers] and wait matrix of the same size

    n = len(customers) - 1
    V = len(vehicles)

    savings = calculate_savings(d)

    unserviced_demands = []
    all_customers_serviced = False

    for customer in customers:
        unserviced_demands.append(customer.demand)

    available_vehicles = vehicles[:]
    fully_serviced = []
    routes = []
    loads = [0 for _ in range(V)]
    f = [[0.0] * (n + 1) for _ in range(V)]

    # Starting route preparation

    # Make restrictions list
    count_R = [0 for _ in range(n)]
    time_window_size = [0 for _ in range(n)]
    demands = unserviced_demands[1:]

    for j in range(n):
        for i in range(V):
            count_R[j] += R[i][j]
        time_window_size[j] = customers[j+1].end_time - customers[j+1].start_time

    # Order candidates for starting route (R > time_window > least demand)
    ranked_indices = sorted(
        range(len(count_R)), 
        key=lambda i: (count_R[i], time_window_size[i], demands[i])
    )
    ranked_indices = [i + 1 for i in ranked_indices]

    # Main loop
    while not all_customers_serviced:
        if len(routes) < V:
            for j in ranked_indices:
                route = []
                for vehicle in available_vehicles:
                    if check_site_dependency(vehicle, j) and check_time_windows(t, [0,j,0], customers):
                        route = [0,j,0]
                        ranked_indices.remove(j)    # Prevent j from starting another route
                        routes.append([route, vehicle.id])
                        available_vehicles.remove(vehicle)
                        loads[vehicle.id] += unserviced_demands[j]
                        last_customer_id = j
                        break
                if len(route) > 0:
                    break
        
        else:   # There is at least one route per vehicle
            for pair in savings:
                i, j = pair              
                route_id = check_routes(pair, routes, fully_serviced)
                
                # Merge pair to the existing routes if possible
                if route_id >= 0:
                    route = routes[route_id][0]
                    vehicle = vehicles[routes[route_id][1]]

                    if loads[vehicle.id] > vehicle.capacity:    # Full vehicle, prevent further routing
                        continue

                    if not (check_site_dependency(vehicle, i) and check_site_dependency(vehicle, j)):
                        continue

                    if (i in route) and (i == route[1] or i == route[len(route) - 2]):
                        route = merge_route(route, pair)
                        if not check_time_windows(t, route, customers):
                            continue
                        routes[route_id][0] = route[:]  # Update route
                        loads[vehicle.id] += unserviced_demands[j] # Update load
                        last_customer_id = j
                        break
                    elif (j in route) and (j == route[1] or j == route[len(route) - 2]):
                        route = merge_route(route, pair)
                        if not check_time_windows(t, route, customers):
                            continue
                        routes[route_id][0] = route[:]  # Update route
                        loads[vehicle.id] += unserviced_demands[i]  # Update load
                        last_customer_id = i
                        break
            else:
                break

        if loads[vehicle.id] > vehicle.capacity:
            serviced_demand = unserviced_demands[last_customer_id] - (loads[vehicle.id] - vehicle.capacity)
            unserviced_demands[last_customer_id] -= serviced_demand
            f[vehicle.id][last_customer_id] = serviced_demand / customers[last_customer_id].demand
        else:
            f[vehicle.id][last_customer_id] = unserviced_demands[last_customer_id] / customers[last_customer_id].demand
            unserviced_demands[last_customer_id] = 0
        
        for i in range(1, len(unserviced_demands)):
            if unserviced_demands[i] < 0.0001 and i not in fully_serviced:
                fully_serviced.append(i)

        for pair in savings[:]:
            i, j = pair
            if (i in fully_serviced) and (j in fully_serviced):
                savings.remove(pair)

        if len(fully_serviced) == n:
            all_customers_serviced = True
    
    # Check and correct concomitances (if necessary)
    concomitances = concomitance_detection(routes, customers, t)
    wait = concomitance_wait(concomitances, routes, customers, t)
    routes, wait = concomitance_correction(wait, concomitances, routes, customers, t)

    return routes, f, wait

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
    # Optimum: v0 = 0-4-5-1(0.2)-0, v1 = 0-3-1(0.8)-2-0, cost = 

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

    final_routes, f, wait = clarke_wright(customers, vehicles, d, t, R)
    
    for i in range(len(final_routes)):
        print(f'Vehicle {final_routes[i][1]} route = {final_routes[i][0]}')
    
    for i in range(len(f)):
        print(f'f[{i}] = {f[i]}') 

    for i in range(len(wait)):
        print(f'wait[{i}] = {wait[i]}')

    print(f'Cost = {calculate_cost(final_routes, d, vehicles)}')

if __name__ == "__main__":
    main()
