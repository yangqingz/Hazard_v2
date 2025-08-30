from src.HAZARD.policy.dstar.priority_queue import PriorityQueue, Priority
from src.HAZARD.policy.dstar.grid import OccupancyGridMap
import numpy as np
from src.HAZARD.policy.dstar.utils import heuristic, Vertex, Vertices
from typing import Dict, List

OBSTACLE = 255
UNOCCUPIED = 0


class DStarLite:
    def __init__(self, map: OccupancyGridMap, s_start: (int, int), s_goal: (int, int)):
        """
        :param map: the ground truth map of the environment provided by gui
        :param s_start: start location
        :param s_goal: end location
        """
        self.new_edges_and_old_costs = None

        # algorithm start
        self.s_start = s_start
        self.s_goal = s_goal
        self.s_last = s_start
        self.k_m = 0  # accumulation
        self.U = PriorityQueue()
        self.rhs = np.ones((map.x_dim, map.y_dim)) * np.inf
        self.g = self.rhs.copy()

        self.sensed_map = OccupancyGridMap(x_dim=map.x_dim,
                                           y_dim=map.y_dim,
                                           exploration_setting='8N')

        self.rhs[self.s_goal] = 0
        self.U.insert(self.s_goal, Priority(heuristic(self.s_start, self.s_goal), 0))

    def calculate_key(self, s: (int, int)):
        """
        :param s: the vertex we want to calculate key
        :return: Priority class of the two keys
        """
        k1 = min(self.g[s], self.rhs[s]) + heuristic(self.s_start, s) + self.k_m
        k2 = min(self.g[s], self.rhs[s])
        return Priority(k1, k2)

    def c(self, u: (int, int), v: (int, int)) -> float:
        """
        calcuclate the cost between nodes
        :param u: from vertex
        :param v: to vertex
        :return: euclidean distance to traverse. inf if obstacle in path
        """
        if not self.sensed_map.is_unoccupied(u) or not self.sensed_map.is_unoccupied(v):
            return float('inf')
        else:
            return heuristic(u, v)

    def contain(self, u: (int, int)) -> (int, int):
        return u in self.U.vertices_in_heap

    def update_vertex(self, u: (int, int)):
        if self.g[u] != self.rhs[u] and self.contain(u):
            self.U.update(u, self.calculate_key(u))
        elif self.g[u] != self.rhs[u] and not self.contain(u):
            self.U.insert(u, self.calculate_key(u))
        elif self.g[u] == self.rhs[u] and self.contain(u):
            self.U.remove(u)

    def compute_shortest_path(self):
        while self.U.top_key() < self.calculate_key(self.s_start) or self.rhs[self.s_start] > self.g[self.s_start]:
            u = self.U.top()
            k_old = self.U.top_key()
            k_new = self.calculate_key(u)

            if k_old < k_new:
                self.U.update(u, k_new)
            elif self.g[u] > self.rhs[u]:
                self.g[u] = self.rhs[u]
                self.U.remove(u)
                pred = self.sensed_map.succ(vertex=u)
                for s in pred:
                    if s != self.s_goal:
                        self.rhs[s] = min(self.rhs[s], self.c(s, u) + self.g[u])
                    self.update_vertex(s)
            else:
                self.g_old = self.g[u]
                self.g[u] = float('inf')
                pred = self.sensed_map.succ(vertex=u)
                pred.append(u)
                for s in pred:
                    if self.rhs[s] == self.c(s, u) + self.g_old:
                        if s != self.s_goal:
                            min_s = float('inf')
                            succ = self.sensed_map.succ(vertex=s)
                            for s_ in succ:
                                temp = self.c(s, s_) + self.g[s_]
                                if min_s > temp:
                                    min_s = temp
                            self.rhs[s] = min_s
                    self.update_vertex(u)

    def rescan(self) -> Vertices:

        new_edges_and_old_costs = self.new_edges_and_old_costs
        self.new_edges_and_old_costs = None
        return new_edges_and_old_costs

    def move_and_replan(self, robot_position: (int, int)):
        path = [robot_position]
        self.s_start = robot_position
        self.s_last = self.s_start
        self.compute_shortest_path()

        while self.s_start != self.s_goal:
            assert (self.rhs[self.s_start] != float('inf')), "There is no known path!"

            succ = self.sensed_map.succ(self.s_start, avoid_obstacles=False)
            min_s = float('inf')
            arg_min = None
            for s_ in succ:
                temp = self.c(self.s_start, s_) + self.g[s_]
                if temp < min_s:
                    min_s = temp
                    arg_min = s_

            ### algorithm sometimes gets stuck here for some reason !!! FIX
            self.s_start = arg_min
            path.append(self.s_start)
            # scan graph for changed costs
            changed_edges_with_old_cost = self.rescan()
            #print("len path: {}".format(len(path)))
            # if any edge costs changed
            if changed_edges_with_old_cost:
                self.k_m += heuristic(self.s_last, self.s_start)
                self.s_last = self.s_start

                # for all directed edges (u,v) with changed edge costs
                vertices = changed_edges_with_old_cost.vertices
                for vertex in vertices:
                    v = vertex.pos
                    succ_v = vertex.edges_and_c_old
                    for u, c_old in succ_v.items():
                        c_new = self.c(u, v)
                        if c_old > c_new:
                            if u != self.s_goal:
                                self.rhs[u] = min(self.rhs[u], self.c(u, v) + self.g[v])
                        elif self.rhs[u] == c_old + self.g[v]:
                            if u != self.s_goal:
                                min_s = float('inf')
                                succ_u = self.sensed_map.succ(vertex=u)
                                for s_ in succ_u:
                                    temp = self.c(u, s_) + self.g[s_]
                                    if min_s > temp:
                                        min_s = temp
                                self.rhs[u] = min_s
                            self.update_vertex(u)
            self.compute_shortest_path()
        print("path found!")
        return path, self.g, self.rhs
    
def get_dstar_weight(sem_map, origin, destination):
    # unexplored: 5, explored: 1
    # height > 0.1: += 10
    # height > 1: += 1000
    explored = sem_map["explored"]
    height = sem_map["height"]
    w, h = explored.shape
    weight = np.ones((w, h))
    weight[explored == 0] = 30
    
    # exp(x * 0.) * y= 10
    # exp(x * 1.6) * y = 1000
    # x = 3, y = 7
    weight[height > 0.5] += np.exp(height[height > 0.5] * 3) * 7
    weight[height > 0.1] += 10
    # weight[height > 0.5] += 50
    # weight[height > 1.6] += 1000
    # for each position, if it is near an obstacle, increase its weight by 50
    conv_weight = np.zeros((w, h))
    for i in range(-2, 3):
        for j in range(-2, 3):
            conv_weight[max(0, i): min(w, w+i), max(0, j): min(h, h+j)] += weight[max(0, -i): min(w, w-i), max(0, -j): min(h, h-j)] * 1.0 / (abs(i) + abs(j) + 1)
    return conv_weight.astype(np.float32)

# if __name__ == "__main__":
#     # Fake semantic map
#     sem_map = {
#         "explored": np.ones((5,5), dtype=np.int32),
#         "height": np.zeros((5,5), dtype=np.float32)
#     }
#     sem_map["explored"][2,2] = 0  # simulate unexplored = obstacle
#     sem_map["height"][3,3] = 1.2  # high wall

#     # Convert to weight grid
#     weight_grid = get_dstar_weight(sem_map, origin=(0,0), destination=(4,4))

#     # Initialize occupancy grid map
#     occ_map = OccupancyGridMap(x_dim=weight_grid.shape[0],
#                                y_dim=weight_grid.shape[1],
#                                exploration_setting='8N')
#     # Mark high cost cells as obstacles for this version
#     occ_map.grid = np.where(weight_grid > 1000, 255, 0).astype(np.uint8)

#     # Create planner and find path
#     planner = DStarLite(occ_map, s_start=(0,0), s_goal=(4,4))
#     path, g, rhs = planner.move_and_replan((0,0))
#     print("Planned Path:", path)