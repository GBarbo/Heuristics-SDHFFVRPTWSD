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

Based on the Clarke-Wright (1964) savings heuristic, the file ```clarke_wright.py``` is my first approach on tackling the SDHFFVRPTWSD with a constructive heursitic. The pseudocode below describes the general procedure.

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
