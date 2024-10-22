import numpy as np
import matplotlib.pyplot as plt

def ant_colony_optimization_tsp():
    # 初始化城市坐标
    cities = np.array([[565.0,575.0],[25.0,185.0],[345.0,750.0],[945.0,685.0],[845.0,655.0],
            [880.0,660.0],[25.0,230.0],[525.0,1000.0],[580.0,1175.0],[650.0,1130.0],
            [1605.0,620.0],[1220.0,580.0],[1465.0,200.0],[1530.0,  5.0],[845.0,680.0],
            [725.0,370.0],[145.0,665.0],[415.0,635.0],[510.0,875.0],[560.0,365.0],
            [300.0,465.0],[520.0,585.0],[480.0,415.0],[835.0,625.0],[975.0,580.0],
            [1215.0,245.0],[1320.0,315.0],[1250.0,400.0],[660.0,180.0],[410.0,250.0],
            [420.0,555.0],[575.0,665.0],[1150.0,1160.0],[700.0,580.0],[685.0,595.0],
            [685.0,610.0],[770.0,610.0],[795.0,645.0],[720.0,635.0],[760.0,650.0],
            [475.0,960.0],[95.0,260.0],[875.0,920.0],[700.0,500.0],[555.0,815.0],
            [830.0,485.0],[1170.0, 65.0],[830.0,610.0],[605.0,625.0],[595.0,360.0],
            [1340.0,725.0],[1740.0,245.0]])
    num_cities = cities.shape[0]  # 城市数量
    num_ants = 20  # 蚂蚁数量
    max_iterations = 100  # 最大迭代次数

    # 初始化参数
    alpha = 1  # 信息素重要程度
    beta = 5  # 启发函数重要程度
    rho = 0.1  # 信息素挥发系数
    Q = 100  # 常量，用于更新信息素

    # 初始化信息素矩阵和距离矩阵
    pheromone = np.ones((num_cities, num_cities))  # 信息素矩阵，初始值为1
    distance = np.full((num_cities, num_cities), np.inf)  # 距离矩阵，初始值为无穷大
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                distance[i, j] = np.linalg.norm(cities[i] - cities[j])  # 计算城市间的欧几里得距离

    # 主循环
    best_length = float('inf')  # 初始化最优路径长度为无穷大
    best_route = []  # 初始化最优路径为空

    plt.figure()  # 创建图形窗口
    for iter in range(max_iterations):
        routes = np.zeros((num_ants, num_cities), dtype=int)  # 存储每只蚂蚁的路径
        route_lengths = np.zeros(num_ants)  # 存储每只蚂蚁路径的长度

        # 每只蚂蚁构建解
        for k in range(num_ants):
            route = np.zeros(num_cities, dtype=int)  # 初始化蚂蚁的路径
            route[0] = np.random.randint(num_cities)  # 随机选择起点
            for step in range(1, num_cities):
                current_city = route[step - 1]  # 当前所在城市
                probabilities = np.zeros(num_cities)  # 初始化选择概率
                for j in range(num_cities):
                    if j not in route[:step]:  # 如果城市未被访问过
                        probabilities[j] = (pheromone[current_city, j] ** alpha) * ((1 / distance[current_city, j]) ** beta)
                probabilities /= probabilities.sum()  # 归一化选择概率
                next_city = np.random.choice(range(num_cities), p=probabilities)  # 根据概率选择下一个城市
                route[step] = next_city  # 更新路径
            routes[k] = route  # 记录蚂蚁的路径
            route_lengths[k] = sum([distance[route[i], route[(i + 1) % num_cities]] for i in range(num_cities)])  # 计算路径总长度

        # 更新最优解
        min_length = route_lengths.min()  # 找到最短路径
        min_index = route_lengths.argmin()  # 找到最短路径的索引
        if min_length < best_length:  # 如果找到更短的路径
            best_length = min_length  # 更新最短路径长度
            best_route = routes[min_index]  # 更新最优路径

        # 更新信息素
        pheromone *= (1 - rho)  # 信息素挥发
        for k in range(num_ants):
            for step in range(num_cities):
                from_city = routes[k, step]
                to_city = routes[k, (step + 1) % num_cities]
                pheromone[from_city, to_city] += Q / route_lengths[k]  # 更新信息素
                pheromone[to_city, from_city] += Q / route_lengths[k]  # 对称更新信息素

        # 绘制当前最优路径
        plt.clf()  # 清除当前图形
        plt.plot(cities[:, 0], cities[:, 1], 'bo')  # 绘制城市
        best_route_cities = np.append(best_route, best_route[0])  # 将最优路径的起点城市添加到路径的末尾，形成环
        for i in range(num_cities):
            from_city = best_route_cities[i]
            to_city = best_route_cities[i + 1]
            plt.arrow(cities[from_city, 0], cities[from_city, 1],
                      cities[to_city, 0] - cities[from_city, 0],
                      cities[to_city, 1] - cities[from_city, 1],
                      color=[0.5, 0.75, 1], head_width=0.2, length_includes_head=True)

        # 标注起点和终点
        start_city = best_route[0]
        end_city = best_route[-1]
        plt.plot(cities[start_city, 0], cities[start_city, 1], 'go', markersize=8, markerfacecolor='g')  # 起点标绿
        plt.plot(cities[end_city, 0], cities[end_city, 1], 'ro', markersize=8, markerfacecolor='r')  # 终点标红
        plt.text(cities[start_city, 0], cities[start_city, 1], '起点', verticalalignment='bottom', horizontalalignment='left')
        plt.text(cities[end_city, 0], cities[end_city, 1], '终点', verticalalignment='bottom', horizontalalignment='left')

        # 标注城市编号
        for i in range(num_cities):
            plt.text(cities[i, 0], cities[i, 1], f'City {i + 1}', verticalalignment='bottom', horizontalalignment='right')

        plt.title(f'Iteration {iter + 1}: Best length = {best_length:.2f}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.pause(0.1)

    # 输出最终结果
    print('Best route:', best_route)
    print(f'Best length: {best_length:.2f}')

if __name__ == '__main__':
    ant_colony_optimization_tsp()