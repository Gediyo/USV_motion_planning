import os
import sys
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../Sampling_based_Planning/")

import env, plotting, utils, queue
import USV_model

np.random.seed(56)  # seeding the random number generator
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
        self.brother = None    # for corresponding nodes on the state/workspace tree

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


class RrtStar:
    def __init__(self, x_start, x_goal, step_len,
                 goal_sample_rate, search_radius, iter_max, USV, U_c):
        self.s_start = Node(x_start)
        self.s_start.brother = self.s_start
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius                  # rewiring radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []                                      # workspace tree path

        self.s_vertex = [self.s_start]                      # state tree vertex
        self.path_stree = []                                # state tree path
        self.Er = 30                                        # radius arounnd the node
        self.USV = USV                                      # USV model
        self.U_c = U_c                                      # water current


        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()

        self.x_range = self.env.x_range
        self.y_range = self.env.y_range
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary

    def planning(self):
        for k in range(self.iter_max):
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if k % 500 == 0: # print the iteration number every 500 iteration
                print("RRT_star iteration: ",k)


            if node_new and not self.utils.is_collision(node_near, node_new):
                expand_stree, State_projection = self.predict_state(node_near, node_new)
                if expand_stree:
                    self.s_vertex += State_projection
                    self.s_vertex[-1].brother = node_new      # assigning a sibling to the last stree node
                    node_new.brother = self.s_vertex[-1]      # assigning a sibling to node_new
                    node_new.cost = self.s_vertex[-1].cost    # assigning cost to new node
                    self.vertex.append(node_new)              # adding node_new to the wtree(vertex)

                    # find neighbouring indices
                    neighbor_index = self.find_near_neighbor(node_new)            

                    if neighbor_index:
                        self.choose_s_parent(node_new, neighbor_index)
                        self.stree_rewire(node_new, neighbor_index)

        index = self.search_goal_parent()
        self.path = self.extract_path(self.vertex[index])

        print("path: extracted")
        self.path_stree = self.extract_trajectory(self.vertex[index].brother)

        print("trajectory: obtained")
        # self.plotting.animation(self.vertex, self.path, "rrt*, N = " + str(self.iter_max))
        self.plotting.animation_SPRRT(self.vertex, self.path_stree, self.s_vertex, "spRRT-star", True)
        self.plotting.animation_SPRRT_star(self.vertex, self.path_stree, self.s_vertex, "spRRT-star", True)

        return self.vertex[index].cost

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

                    return True, State_projection
                    
        return False, []

    # def corresonding_stree_index(self, neighbor_index):
    #     stree_neighbor_index = [self.s_vertex.index(self.vertex[i].brother) for i in neighbor_index]
    #     return stree_neighbor_index

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def choose_s_parent(self, node_new, neighbor_index):
        # cost = [self.s_vertex[i].cost for i in neighbor_s_index]
        # cost_min_index = neighbor_s_index[int(np.argmin(cost))]

        # sorted_snodes = sorted(neighbor_s_index, key = lambda x: self.s_vertex[x].cost)
        sorted_nodes = sorted(neighbor_index, key = lambda x: self.vertex[x].cost)

        for index in sorted_nodes:
            if index == self.vertex.index(node_new) or index == self.vertex.index(node_new.parent):
                continue

            path_exist, node_list = self.predict_state(self.vertex[index], node_new)

            if path_exist:
                self.s_vertex += node_list # add the node list to the state tree
                node_new.parent = self.vertex[index]
                node_new.brother = self.s_vertex[-1]
                node_new.cost = self.s_vertex[-1].cost
                break

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def stree_rewire(self, node_new, neighbor_index):
        for i in neighbor_index:

            if i == self.vertex.index(node_new) or i == self.vertex.index(node_new.parent):
                continue

            node_neighbor = self.vertex[i]
            
            path_exist, node_list = self.predict_state(node_new, node_neighbor)

            # if path exists and neighbor node cost higher than the cost to get there through node new
            if path_exist and node_neighbor.cost > node_list[-1].cost:
                node_neighbor.parent = node_new             # assign parent of the neighbouring node to node_new
                self.s_vertex += node_list                  # add the node_list to stree
                node_neighbor.brother = self.s_vertex[-1]   #assign the sibling to be the last of node list
                self.s_vertex[-1].brother = node_neighbor   # update sibling on the wtree
                node_neighbor.cost = self.s_vertex[-1].cost # update cost

    def search_goal_parent(self):
        # caculates dist between the workspace tree and the goal node
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex] 

        # index of nodes from the workspace tree that are within step distance from the goal
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.Er]

        if len(node_index) > 0:
            # cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
            #              if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            cost_s_list = [self.vertex[i].cost for i in node_index]

            return node_index[int(np.argmin(cost_s_list))] # returns based on lower cost of the state tree

        return len(self.vertex) - 1    # or return the last node 

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1            # the total number of vertices,  incremented by 1 to account for the new node.
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len) # min. b/n the log term which decreases as the no. of nodes increase or the step size

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]

        # indices of nearby collision free neighbors
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]

        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent

        return cost

    def update_cost(self, parent_node):
        OPEN = queue.QueueFIFO()
        OPEN.put(parent_node)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.child) == 0:
                continue

            for node_c in node.child:
                node_c.Cost = self.get_new_cost(node, node_c)
                OPEN.put(node_c)

    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        node = node_end

        while node.parent is not None:
            print("index appending to path:", self.vertex.index(node))
            path.append([node.x, node.y])
            node = node.parent
        path.append([node.x, node.y])

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


def main():
    
    x_start = (2, 2, 0)  # Starting node
    x_goal = (4700, 2500, np.pi)  # Goal node

    U_c = [0, 0]

    USV = USV_model.frigate(U = 5, r = 0 )  # r = 0 represent that the vessile can't rotate in place
    
    rrt_star = RrtStar(x_start, x_goal, 450, 0.10, 1000, 10000, USV, U_c)
    
    # x_start = (18, 8)  # Starting node
    # x_goal = (37, 18)  # Goal node

    # rrt_star = RrtStar(x_start, x_goal, 10, 0.10, 20, 10000)
    rrt_star.planning()


if __name__ == '__main__':
    main()