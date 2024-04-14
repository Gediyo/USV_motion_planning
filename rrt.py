import os
import sys
import math
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

import env, plotting, utils
import USV_model

# np.random.seed(6) # seeding a random number generator

class Node:
    def __init__(self, n):

        """
            RRT Node

            Arribute
            ----------------
            n   - pose of the vessile
        
        """

        self.x = n[0]
        self.y = n[1]
        self.yaw = 0
        self.cost = 0
        self.path_x = []
        self.path_y = []  
        self.path_yaw = []
        self.parent = None
        self.brother = None

        if len(n) == 3:
            self.yaw = n[2]
            

    def copy(self):
        # Create a new Node object with the same data
        new_node = Node((self.x, self.y, self.yaw))
        new_node.cost = self.cost
        new_node.path_x = self.path_x.copy()
        new_node.path_y = self.path_y.copy()
        new_node.path_yaw = self.path_yaw.copy()
        new_node.parent = self.parent
        new_node.brother = self.brother
        
        # If you have other attributes in your Node class, you may need to copy them as well
        return new_node
    
    def is_state_identical(self, node):
        """
            check if x, y, yaw are the identical
        """
        if abs(node.x - self.x) > 0.001:
            return False
        elif abs(node.y - self.y) > 0.001:
            return False
        elif abs(node.yaw - self.yaw) > 0.001:
            return False

        return True

    def print_node(self):
        """
            Node printing function. USE FOR DEBUGGING.
        """
        print(f'x: {self.x}, y:{self.y}, yaw: {self.yaw}')


class Rrt:
    def __init__(self, s_start, s_goal, step_len, goal_sample_rate, iter_max, USV, U_c):
        self.s_start = Node(s_start)
        self.s_start.brother = self.s_start
        self.s_goal = Node(s_goal)
        self.step_len = step_len                            # step length towards the new sampled node
        self.goal_sample_rate = goal_sample_rate            # probablity to sample the goal node
        self.iter_max = iter_max                            # maximum iteration
        self.vertex = [self.s_start]                        # list of nodes

        self.s_vertex = [self.s_start]                      # state tree vertex
        self.Er = 30                                      # radius arounnd the node
        self.USV = USV                                      # USV model
        self.U_c = U_c

        self.env = env.Env()                                # imprting map
        self.plotting = plotting.Plotting(s_start, s_goal)  # plot start and end goal
        self.utils = utils.Utils()

        self.x_range = self.env.x_range                     # map range on x
        self.y_range = self.env.y_range                     # map range on y
        self.obs_circle = self.env.obs_circle               # circular obstacle
        self.obs_rectangle = self.env.obs_rectangle         # rectangular obstacle
        self.obs_boundary = self.env.obs_boundary           # boundary obstacle


    def planning(self):
        
        for i in range(self.iter_max):

            if i % 500 == 0:
                print("RRT iteration: ",i)

            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)
            

            if node_new and not self.utils.is_collision(node_near, node_new):
                expand_stree = self.predict_state(node_near, node_new)
                if expand_stree:
                    self.s_vertex[-1].brother = node_new
                    node_new.brother = self.s_vertex[-1]
                    node_new.cost = self.s_vertex[-1].cost
                    self.vertex.append(node_new)
                    
                    dist, _ = self.get_distance_and_angle(node_new, self.s_goal)

                    if dist <= self.step_len and not self.utils.is_collision(node_new, self.s_goal):
                        expand_final_stree = self.predict_state(node_new, self.s_goal)
                        if expand_final_stree:
                            last_node = self.new_state(node_new, self.s_goal)
                            self.s_vertex[-1].brother = last_node
                            last_node.brother = self.s_vertex[-1]
                            last_node.cost = self.s_vertex[-1].cost
                            return self.extract_path(node_new), self.extract_trajectory(self.s_vertex[-1]), last_node.cost
                        

        return None, None, None
    

    
    def predict_state(self, node_near, node_new):

        dt = 5 # time in seconds
        
        # list of rudder angles
        rudder_angles = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

        node_s_near = node_near.brother #find the state tree node assiciated with the near node
        
        
        for rudder_angle in rudder_angles:

            new_node = node_s_near.copy()
                                
            rudder_angle = rudder_angle * np.pi/180 # convert to radians            
            
            new_nu = self.USV.dynamics(self.USV.nu, rudder_angle, dt)

            U = new_nu[0]  # linear velocity of the vessel
            r = new_nu[5]  # angular velocity of the vessel

            for t in range (1, 21):
                State_projection = []
                old_node = new_node


                yaw = new_node.yaw + (dt * r)
                x = new_node.x + (dt * U * np.cos(new_node.yaw))
                y = new_node.y + (dt * U * np.sin(new_node.yaw))

                x += self.U_c[0] * dt
                y += self.U_c[1] * dt

                new_node = Node([x,y,yaw])
                new_node.parent = old_node
                new_node.cost = old_node.cost + dt


                State_projection.append(new_node)

                if self.utils.is_collision(old_node, new_node):
                    
                    break

                if math.hypot(new_node.x - node_new.x, new_node.y - node_new.y) <= self.Er:
                    self.s_vertex += State_projection

                    return True
                    
        return False

    def generate_random_node(self, goal_sample_rate):
        # minimum distance the vessile need to keep 
        #from obstacles and the edge of the map
        delta = self.utils.delta   
        
        # return goal node if the random number generated is greater than the goal_sample_rate
        # biased sampling technique
        if np.random.random() > goal_sample_rate: 
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    def new_state(self, node_start, node_end):
        dist, theta = self.get_distance_and_angle(node_start, node_end)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))
        node_new.parent = node_start

        return node_new

    def extract_path(self, node_end):
        path = [(self.s_goal.x, self.s_goal.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path
    
    def extract_trajectory(self, node_end):
        path = [(node_end.x, node_end.y)]
        node_now = node_end

        while node_now.parent is not None:
            node_now = node_now.parent
            path.append((node_now.x, node_now.y))

        return path

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    # Wraps angle to [-pi,pi] range
    def wraptopi(x):
        if x > np.pi:
            x = x - (np.floor(x / (2 * np.pi)) + 1) * 2 * np.pi
        elif x < -np.pi:
            x = x + (np.floor(x / (-2 * np.pi)) + 1) * 2 * np.pi
        return x


def main():
    x_start = (-2, 2, 0)  # Starting node
    x_goal = (4000, 4000, np.pi)  # Goal node

    U_c = [0, 2]

    USV = USV_model.frigate(U = 5, r = 0 )  # r = 0 represent that the vessile can't rotate in place

    rrt = Rrt(x_start, x_goal, 450, 0.1, 10000, USV, U_c)

    path_wtree, path_stree, _ = rrt.planning()

    if path_wtree:
        # rrt.plotting.animation(rrt.vertex, path_wtree, "RRT", True)
        # rrt.plotting.animation(rrt.s_vertex, path_stree, "spRRT", True)
        rrt.plotting.animation_SPRRT(rrt.vertex, path_stree, rrt.s_vertex, "spRRT", True)

    else:
        print("No Path Found!")

if __name__ == '__main__':
    main()