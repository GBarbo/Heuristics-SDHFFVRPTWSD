# Heuristics-SDHFFVRPTWSD
Heuristics developed to tackle the Site Dependent Heterogeneous Fixed Fleet Vehicle Routing Problem with Time Windows and Split Deliveries. 

## Usage

Simply run the ```.py``` files with the algorithm of choice or import their main functions with the same name as the files. Make sure all the following parameters are available:

Classes: <br>
- ```vehicles```: must contain capacity ```a``` (float), freight cost ```cf``` (float) and site dependency matrix (```R```[vehicles x customers] (int)).
- ```customers```: must contain demand ```q``` (float), minimum start time ```e``` (float), service time ```s``` (float) and maximum ending time ```l``` (float).

Matrices: <br>
- distances ```d```: matrix [customers x customers] of distances between two customers or depot (int).
- travel time ```t```: matrix [customers x customers] of travel time between two customers or depot (float).

## Algorithms

### Clarke-Wright for the SDHFFVRPTWSD

Based on the Clarke-Wright (1964) savings heuristic, the files ```clarke_wright_series.py``` and ```clarke_wright_paralell.py``` are my first approaches on tackling the SDHFFVRPTWSD with a constructive heursitic. The ```clarke_wright.py``` file has the best approach tailored to tackle the problem.

#### Series Clarke-Wright
This algorithm iteratively merges routes until the vehicle capacity is exceeded; only then, another route is created if there are available vehicles. Thus, it is best suited for instances which the freight cost spread is high and it is very important to get the best possible route to the vehicles with least freight cost.
The pseudocode below describes the general procedure.

```text
1.  calculate savings for each pair of clients
2.  sort savings in descending order of saving
3.  routed_customers <- Ø
4.  final_routes <- Ø
5.  WHILE there are available vehicles DO:
6.      route <- Ø
7.      load = 0
8.      vehicle = available vehicle with the least freight cost
9.      WHILE possible DO:
10.         IF route is empty:
11.             start route with feasible candidate from the best unrouted savings pairs
12.            IF there is no feasible candidate:
13.                BREAK WHILE
14.            ELSE:
15.                 load += unserviced demand of the last customer
16.                 set last customer = candidate
17.             ENDIF
18.         ELSE:
19.             FOR (i, j) in sorted savings pairs DO:
20.                 IF either i or j are either the first or last customers and the new route is feasible:
21.                     merge route with [0,i,0] or [0,j,0]
22.                     load += unserviced demand of of the last customer
23.                     set last customer = i or j
24.                     BREAK FOR
25.                 ELSE:
26.                     CONTINUE
27.             ELSE:
28.                 BREAK WHILE
29.             ENDFOR
30.         ENDIF
31.         IF load > vehicle capacity:
32.             serviced demand = unserviced demand - (load - vehicle capacity)
33.             unserviced demand -= serviced demand
34.             f[vehicle][last customer] = serviced demand / total demand
35.             BREAK WHILE
36.         ELSE:
37.             f[vehicle][last customer] = unserviced demand / total demand
38.             unserviced demand = 0
39.         ENDIF
40.     ENDWHILE
41.     FOR each customer DO:
42.         IF unserviced demand of the customer == 0:
43.             add customer to the routed_customers
44.         ENDIF
45.     ENDFOR
46.     add route to final routes and assign the vehicle for it
47.     remove the vehicle from available vehicles
48. ENDWHILE
49. RETURN final routes and f matrix
```

#### Paralell Clarke-Wright
This algorithm iteratively merges single customers with the existing routes and creates new routes when the merge is not possible with the existing ones. The objective is to make best use of the highest savings pair of customers. The pseudocode below describes the general procedure.

```text
1. calculate savings for each pair of clients
2. sort savings in descending order of saving
3. unserviced_demands <- Ø
4. fully_serviced <- Ø
5. routes <- Ø
6. available_vehicles <- vehicles
7. WHILE there are customers with unserviced demands DO:
8.     FOR (i, j) in sorted savings pairs DO:
9.         IF there are available_vehicles without a route:
10.            IF either i or j belongs to a route and both not to the others:
11.                route = route from routes containing either i or j
12.                vehicle = vehicle for which the route was assigned
13.                IF the vehicle is full:
14.                    CONTINUE
15.                ENDIF
16.                IF either i or j are either the first or last customers and the new route is feasible:
17.                    merge route with [0,i,0] or [0,j,0]
18.                    update route in the routes list
19.                    load[vehicle] += unserviced demand of the merged customer
20.                    set last customer = merged customer
21.                    BREAK FOR
22.                ENDIF
23.            ELSE IF neither i nor j belong to an existing route:
24.                vehicle = available vehicle with minimum freight cost
25.                start route with either i or j
26.                IF neither is feasible:
27.                    CONTINUE
28.                ENDIF
29.                remove vehicle from available_vehicles
30.                add route with vehicle to the routes list
31.                load[vehicle] += unserviced demand of the feasible candidate
32.                set last customer = feasible candidate
33.                BREAK FOR
34.            ENDIF
35.        ELSE:
36.            IF either i or j belongs to a route and both not to the others:
37.                same merging procedure as when there were available_vehicles
38.            ENDIF
39.        ENDIF 
40.    ELSE:
41.        BREAK WHILE
42.    ENDFOR
43.    IF load[vehicle] > vehicle capacity:
44.        serviced demand = unserviced demand[last customer] - (load[vehicle] - vehicle capacity)
45.        unserviced demand[last customer] -= serviced demand
46.        f[vehicle][last customer] = serviced demand / total demand
47.        mark vehicle as full so it cannot be routed anymore
48.    ELSE:
49.        f[vehicle][last customer] = unserviced demand / total demand
50.        unserviced demand[last customer] = 0
51.    ENDIF
52. ENDWHILE
53. RETURN routes and f matrix

```

#### Paralell Clarke-Wright with Starting Criterium

Previous paralell Clarke-Wright with a restriction criterium to start route: customers that can be serviced with less vehicles are chosen to start the routes for the vehicles that can service them. If there is a draw, the customer with a smaller time window is routed first. If there is another draw, the criterium takes the customer with the least demand (so its service does not affect significately the vehicle's capacity). The pseudocode below describes the general procedure.

```text
calculate savings for each pair of clients
sort savings in descending order of saving
unserviced_demands <- Ø
fully_serviced <- Ø
routes <- Ø
available_vehicles <- vehicles
WHILE there are customers with unserviced demands DO:
    IF there is at least a vehicle without a route assigned to it:
        chose the customer j with most restrictions (R, time windows, least demand)
        IF the route [0,j,0] is feasible for the vehicle that can service j:
            IF the vehicle was not yet assigned to a route:
                add [0,j,0] to routes and assign the vehicle for it
            ENDIF
        ENDIF
    ELSE:
        FOR (i, j) in sorted savings pairs DO:
            IF either i or j belongs to a route and both not to the others:
                route = route from routes containing either i or j
                vehicle = vehicle for which the route was assigned
                IF the vehicle is full:
                    CONTINUE
                ENDIF
                IF either i or j are either the first or last customers and the new route is feasible:
                    merge route with [0,i,0] or [0,j,0]
                    update route in the routes list
                    load[vehicle] += unserviced demand of the merged customer
                    set last customer = merged customer
                    BREAK FOR
                ENDIF
            ENDIF
        ELSE:
            BREAK WHILE
        ENDFOR
    ENDIF
    IF load[vehicle] > vehicle capacity:
        serviced demand = unserviced demand[last customer] - (load[vehicle] - vehicle capacity)
        unserviced demand[last customer] -= serviced demand
        f[vehicle][last customer] = serviced demand / total demand
        mark vehicle as full so it cannot be routed anymore
    ELSE:
        f[vehicle][last customer] = unserviced demand / total demand
        unserviced demand[last customer] = 0
    ENDIF
ENDWHILE
RETURN routes and f matrix
```

#### Clarke Wright GRASP

An additional Clarke-Wright algorithm is used to generate multiple solutions. The ```clarke_wright_grasp``` function generates a population of feasible and infeasible (for time-windows or site depedency) solutions. To do that, instead of picking the best pair according to savings, a random pair is chosen according to the harmonic distribution, so that pairs with greater saving have a probability of being picked two times greater than the next pair of savings.
