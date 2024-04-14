import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

import env, plotting, utils, queue
import USV_model, rrt_star, rrt



def main():
    print("Executing: " + __file__)

    # Set Initial parameters
    
    rrt_star_cost = []
    rrt_cost = []
    i_num = 2

    for i in range(i_num):
        # start_x = np.random.uniform(1.5,4 )
        # start_y = np.random.uniform(2, 18)
        # start_yaw = np.random.uniform(0, 2*np.pi)
    

        # goal_x = np.random.uniform(16,18.5)
        # goal_y = np.random.uniform(2, 18)
        # goal_yaw = np.random.uniform(0, 2*np.pi)

        # start and end goal node instances of sn object
        start = [2, 2, 0 ]
        goal = [4000, 4000, np.pi]

        # ocean current
        U_c = [0, 2]

        # USV model
        USV = USV_model.frigate(U = 5, r = 0 )  # r = 0 represent that the vessile can't rotate in place

        # rrt* algorithm
        RRT_star = rrt_star.RrtStar(start, goal, 450, 0.10, 1000, 10000, USV, U_c)
        RRT_star_cost = RRT_star.planning()

        # saving the result for rrt* algorithm
        if RRT_star_cost is not None:
            rrt_star_cost.append(RRT_star_cost)
        else:
            rrt_star_cost.append(0)

        # rrt algorithm
        RRT = rrt.Rrt(start, goal, 450, 0.1, 10000, USV, U_c)

        _ ,_, RRT_cost = RRT.planning()
        
        #saving the result for RRT algorithm
        if RRT_cost is not None:
            rrt_cost.append(RRT_cost)
        else:
            rrt_cost.append(0)
        
        print('Iteration:', i )

    def calculate_average(numbers):
        total = sum(numbers)
        count = len(numbers)
        if count == 0:
            return 0  # To avoid division by zero if the list is empty
        else:
            return total / count
        
    def standard_deviation(numbers):
        # Calculate the mean
        mean = sum(numbers) / len(numbers)
        
        # Calculate the squared differences from the mean
        squared_diff = [(x - mean) ** 2 for x in numbers]
        
        # Calculate the variance
        variance = sum(squared_diff) / len(numbers)
        
        # Calculate the standard deviation
        std_dev = variance ** 0.5
        
        return std_dev
    
    rrt_star_avg_path = calculate_average(rrt_star_cost)
    rrt_avg_path = calculate_average(rrt_cost)

    rrt_star_std = standard_deviation(rrt_star_cost)
    rrt_std = standard_deviation(rrt_cost)

    print("RRT_star path standard deviation: ", rrt_star_std)
    print("RRT path standard deviation: ", rrt_std)

    plt.figure()
    x = ['mean RRT path', 'mean RRT* path' ]
    y = [rrt_avg_path , rrt_star_avg_path]
    # Plotting the bars
    plt.bar(x, y, width = 0.5, align="center")
    # Adding labels and title
    plt.xlabel('Algorithm')
    plt.ylabel('Average path length (cost)')
    plt.title('Average path of RRT vs RRT* Algorithm')

    # Display the plot
    plt.show()



if __name__ == '__main__':
    main()