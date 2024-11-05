import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import networkx as nx

def bankproblem():
    filename = 'BankProblem.txt'
    data = []

    with open(filename, 'r') as f:
        lines = f.readlines()
        first_line = lines[0].strip()

        #Extracting parts of numbers using regular expressions
        match = re.search(r'security van capacity: (\d+)', first_line) #Match security van capacity: followed by an integer
        if match:
            weight_limit = int(match.group(1))
        else:
            raise ValueError("weight error!!")

        for i in range(1, len(lines), 3):
            bag_match = re.search(r'bag (\d+):', lines[i])#Match the number after the bag
            if bag_match:
                bag = int(bag_match.group(1))
            else:
                continue

            weight_match = re.search(r'weight: ([\d.]+)', lines[i + 1])#Match the number after the weight
            if weight_match:
                weight = float(weight_match.group(1))
            else:
                continue

            value_match = re.search(r'value: (\d+)', lines[i + 2])#Match the number after the value
            if value_match:
                value = int(value_match.group(1))
            else:
                continue

            data.append([bag, weight, value])#Add bag, weight, value to the list data

    df = pd.DataFrame(data, columns=["Bag", "Weight", "Value"])#Convert data to dataframe format via pandas
    print("Vehicle weight limit:", weight_limit)

    num_bags = df.shape[0]
    num_ants = 10
    max_iterations = 10

    best_value_all= []

    # Initialisation parameters
    alpha = 1 #Importance weighting of pheromones
    beta = 5 #Importance weighting of heuristic information
    rho = 0.1 #Pheromone dissipation factor
    Q = 100 #Pheromone intensity

    pheromone = np.ones((num_bags, num_bags)) #pheromone[i][j]: indicates the concentration of pheromone remaining on the path from bag i to bag j.
    
    best_value = 0
    best_route = []

    for iter in range(max_iterations):
        routes = []  # Store the solution for each ant
        route_values = []  # Store the value of each ant solution
        for k in range(num_ants):
            weight_now = 0
            value_now = 0
            route = []
            while True:
                items = [i for i in range(num_bags) if i not in route and (weight_now + df['Weight'].iloc[i] <= weight_limit)] #Selection of viable bags and judgement of the need to stop selection

                if not items:
                    break

                # Calculate the heuristic function (η) and the selection probability
                eta = [df['Value'].iloc[i] / df['Weight'].iloc[i] for i in items]  # Heuristic function: value per unit weight
                pheromone_values = [pheromone[route[-1], i] for i in items] if route else [1] * len(items) #Calculate the list of pheromone values for the current selectable bag


                # Calculate the probability of each bags being selected
                probabilities = [(pheromone_values[i] ** alpha) * (eta[i] ** beta) for i in range(len(items))]
                probabilities_sum = sum(probabilities)
                probabilities = [p / probabilities_sum for p in probabilities]

                # Using roulette to select items
                selected_item = np.random.choice(items, p=probabilities)

                # update solution
                route.append(selected_item)
                weight_now += df['Weight'].iloc[selected_item]
                value_now += df['Value'].iloc[selected_item]

            routes.append(route)
            route_values.append(value_now)

            # update best solution
            if value_now > best_value:
                best_value = value_now
                best_route = route

        # pheromone volatilisation
        pheromone = (1 - rho) * pheromone

        # Pheromone update (based on solutions found by all ants)
        for k in range(num_ants):
            for i in range(len(routes[k]) - 1):
                pheromone[routes[k][i], routes[k][i + 1]] += Q / route_values[k]

        # Print the current optimal solution
        print(f"Iteration {iter + 1}: Best Value = {best_value}")

        best_value_all.append(best_value)

    # Output the final result
    print("Best Route (bags):", [df['Bag'].iloc[i] for i in best_route])
    print("Best Value:", best_value)


    #drawing
    plt.plot(range(1,max_iterations + 1),best_value_all,marker = 'o')
    plt.xlabel('Iterations')
    plt.ylabel('Best Values')
    plt.grid(True)
    plt.show()


    G = nx.DiGraph()
    edges = [(best_route[i], best_route[i + 1]) for i in range(len(best_route) - 1)]
    G.add_edges_from(edges)
    start_node = best_route[0]
    end_node = best_route[-1]



    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_color='black')
    plt.title('Best Path')
    plt.show()



if __name__ == '__main__':
    bankproblem()
