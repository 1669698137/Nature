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
        # 使用正则表达式提取数字部分
        match = re.search(r'security van capacity: (\d+)', first_line)
        if match:
            weight_limit = int(match.group(1))
        else:
            raise ValueError("weight error!!")

        for i in range(1, len(lines), 3):
            bag_match = re.search(r'bag (\d+):', lines[i])
            if bag_match:
                bag = int(bag_match.group(1))
            else:
                continue

            weight_match = re.search(r'weight: ([\d.]+)', lines[i + 1])
            if weight_match:
                weight = float(weight_match.group(1))
            else:
                continue

            value_match = re.search(r'value: (\d+)', lines[i + 2])
            if value_match:
                value = int(value_match.group(1))
            else:
                continue

            data.append([bag, weight, value])

    df = pd.DataFrame(data, columns=["Bag", "Weight", "Value"])
    print("车辆重量限制:", weight_limit)

    num_bags = df.shape[0]
    num_ants = 10
    max_iterations = 10

    best_value_all= []

    # 初始化参数
    alpha = 1
    beta = 5
    rho = 0.1
    Q = 100

    pheromone = np.ones((num_bags, num_bags))
    
    best_value = 0
    best_route = []

    for iter in range(max_iterations):
        routes = []  # 存储每只蚂蚁的解
        route_values = []  # 存储每只蚂蚁解的价值
        for k in range(num_ants):
            weight_now = 0
            value_now = 0
            route = []
            while True:
                items = [i for i in range(num_bags) if i not in route and (weight_now + df['Weight'].iloc[i] <= weight_limit)]

                if not items:
                    break

                # 计算启发函数 (η) 和选择概率
                eta = [df['Value'].iloc[i] / df['Weight'].iloc[i] for i in items]  # 启发函数：单位重量价值
                pheromone_values = [pheromone[route[-1] if route else 0, i] for i in items]

                # 计算每个物品被选择的概率
                probabilities = [(pheromone_values[i] ** alpha) * (eta[i] ** beta) for i in range(len(items))]
                probabilities_sum = sum(probabilities)
                probabilities = [p / probabilities_sum for p in probabilities]

                # 使用轮盘赌法选择物品
                selected_item = np.random.choice(items, p=probabilities)

                # 更新当前解
                route.append(selected_item)
                weight_now += df['Weight'].iloc[selected_item]
                value_now += df['Value'].iloc[selected_item]

            routes.append(route)
            route_values.append(value_now)

            # 更新最优解
            if value_now > best_value:
                best_value = value_now
                best_route = route

        # 信息素挥发
        pheromone = (1 - rho) * pheromone

        # 信息素更新（根据所有蚂蚁找到的解）
        for k in range(num_ants):
            for item in routes[k]:
                pheromone[item, item] += Q / route_values[k]

        # 打印当前最优解
        print(f"Iteration {iter + 1}: Best Value = {best_value}")

        best_value_all.append(best_value)

    # 输出最终结果
    print("Best Route (bags):", [df['Bag'].iloc[i] for i in best_route])
    print("Best Value:", best_value)


    #绘图
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


    # 绘制最佳路径
    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(G)
    nx.draw_networkx(G, pos, with_labels=True, node_size=300, node_color='lightblue', font_color='black')
    plt.title('Best Path')
    plt.show()



if __name__ == '__main__':
    bankproblem()
