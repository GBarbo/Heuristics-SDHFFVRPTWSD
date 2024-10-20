# ==================================================
# Clarke-Wright Algorithm for the SDHFFVRPTWSD
# Author: Giovanni Cesar Meira Barboza
# Date: 2024-10-14
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

    feasible = True
    time = customers[0].start_time
    for i in range(1, len(possible_route)):
        time += t[possible_route[i-1]][possible_route[i]]
        if time < customers[possible_route[i]].start_time:
            feasible = False
            #print(f'start time violation at {possible_route[i]}')
        time += customers[possible_route[i]].service_time
        if time > customers[possible_route[i]].end_time:
            feasible = False
            #print(f'end time violation at {possible_route[i]}')
            
    return feasible

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

def clarke_wright(customers, vehicles, d, t):
    # Input: list of customers, list of vehicles, matrix of distances (d) and matrix of travel times (t)
    # Output: Feasible routes for each vehicle and matrix of split deliveries f [vehicles x customers]

    n = len(customers) - 1
    V = len(vehicles)

    savings = calculate_savings(d)

    unserviced_demands = []
    for customer in customers:
        unserviced_demands.append(customer.demand)
    
    available_vehicles = vehicles[:]
    routed_customers = []
    final_routes = []
    f = [[0.0] * (n + 1) for _ in range(V)]

    while len(available_vehicles) > 0:

        # Pick the available vehicle with minimum cost
        vehicle = min(available_vehicles, key=lambda vehicle: vehicle.freight_cost)

        # Start with an empty route and an empty load
        route = []
        load = 0

        while True:
            # Starting route
            if len(route) == 0:
                for pair in savings:
                    i, j = pair
                    if (i not in routed_customers) and (j not in routed_customers):
                        if check_time_windows(t, [0,i,0], customers) and check_site_dependency(vehicle, i):
                            route = [0,i,0]
                            last_customer = i
                            load += unserviced_demands[i]
                            break
                        elif check_time_windows(t, [0,j,0], customers) and check_site_dependency(vehicle, j):
                            route = [0,j,0]
                            last_customer = j
                            load += unserviced_demands[j]
                            break
                else:
                    break

            # Route already started
            else:
                for pair in savings:
                    i, j = pair
                    if (i in route) and (j in route):
                        continue
                    if (i in routed_customers) or (j in routed_customers):
                        continue
                    if i in route and check_site_dependency(vehicle, j):
                        if route.index(i) not in [1,len(route) - 2]:
                            continue
                        new_route = merge_route(route, (i, j))
                        if check_time_windows(t, new_route, customers):
                            route = new_route[:]
                            load += unserviced_demands[j]
                            last_customer = j
                            break
                    elif j in route and check_site_dependency(vehicle, i):
                        if route.index(j) not in [1,len(route)-2]:
                            continue
                        new_route = merge_route(route, (i, j))
                        if check_time_windows(t, new_route, customers):
                            route = new_route[:]
                            load += unserviced_demands[i]
                            last_customer = i
                            break
                else:
                    break
            if load > vehicle.capacity:
                serviced_demand = unserviced_demands[last_customer] - (load - vehicle.capacity)
                unserviced_demands[last_customer] -= serviced_demand
                f[vehicle.id][last_customer] = serviced_demand / customers[last_customer].demand
                break
            else:
                f[vehicle.id][last_customer] = unserviced_demands[last_customer] / customers[last_customer].demand
                unserviced_demands[last_customer] = 0
        
        for i in range(1, len(unserviced_demands)):
            if unserviced_demands[i] < 0.0001:
                routed_customers.append(i)

        final_routes.append([route, vehicle.id])
        available_vehicles.remove(vehicle)

    return final_routes, f

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
    l = [24, 20, 20, 14, 12, 13]    # End time (hours)

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

    final_routes, f = clarke_wright(customers, vehicles, d, t)
    
    for i in range(len(final_routes)):
        print(f'Vehicle {final_routes[i][1]} route = {final_routes[i][0]}')
    
    for i in range(len(f)):
        print(f'f[{i}] = {f[i]}') 

if __name__ == "__main__":
    main()
