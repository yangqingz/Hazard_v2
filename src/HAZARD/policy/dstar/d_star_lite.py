from src.HAZARD.policy.dstar.priority_queue import PriorityQueue, Priority
from src.HAZARD.policy.dstar.grid import OccupancyGridMap
import numpy as np
import matplotlib.pyplot as plt
from src.HAZARD.policy.dstar.utils import heuristic, Vertex, Vertices
from typing import Dict, List

try:
    from scipy import ndimage as ndi
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False
    
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

    
def get_dstar_weight(sem_map, origin=None, destination=None, inflation_radius=3, obstacle_threshold=0.3):
    """
    Build a traversability / cost map that strongly penalizes cells near obstacles
    and accounts for height. Returns a float32 weight array same shape as sem_map["explored"].
    - inflation_radius: how many cells around obstacle to inflate
    - obstacle_threshold: height value above which a cell is considered an obstacle
    """
    explored = sem_map["explored"]
    height = sem_map["height"]
    if explored is None or height is None:
        raise ValueError("sem_map must contain 'explored' and 'height'")

    w, h = explored.shape
    # base weight: free vs unexplored
    weight = np.ones((w, h), dtype=np.float32)
    weight[explored == 0] = 30.0  # unexplored penalty

    # add height-related cost (gentle slope penalty)
    weight += (height * 4.0).astype(np.float32)
    # mark high obstacles as effectively blocked
    weight[height > 1.6] = 1e6

    # obstacle mask: cells considered obstacles for inflation (darker gray in your image)
    obstacle_mask = height >= obstacle_threshold

    # distance to nearest obstacle (for each free cell)
    if _HAS_SCIPY:
        # distance from free cells to nearest obstacle
        dist = ndi.distance_transform_edt(~obstacle_mask)
    else:
        # fallback: simple convolution-based distance approximation
        dist = np.full((w, h), np.inf, dtype=np.float32)
        obstacle_idx = np.argwhere(obstacle_mask)
        if obstacle_idx.size == 0:
            dist[:] = np.inf
        else:
            coords = np.indices((w, h)).transpose(1, 2, 0).reshape(-1, 2)
            coords = coords.reshape(w * h, 2)
            # compute vectorized Euclidean distance (might be slow for big maps)
            ox = obstacle_idx[:, 0][:, None]
            oy = obstacle_idx[:, 1][:, None]
            cx = np.arange(w)[:, None]
            cy = np.arange(h)[None, :]
            # approximate using broadcasting in chunks to avoid memory blow-up
            for i in range(w):
                for j in range(h):
                    if obstacle_mask[i, j]:
                        dist[i, j] = 0.0
            # coarse fallback: Manhattan-ish propagation
            for r in range(1, inflation_radius + 10):
                mask = (np.abs(np.arange(w)[:, None] - np.arange(w)[None, :]) + np.abs(np.arange(h)[None, :] - np.arange(h)[:, None]))  # not ideal but safe
                break
            dist[np.isinf(dist)] = inflation_radius + 5.0

    # penalty: cells within inflation_radius get growing cost; farther cells small or none
    penalty = np.zeros_like(weight, dtype=np.float32)
    # avoid divide-by-zero; cap distances
    capped = np.minimum(dist, inflation_radius)
    inside = capped < inflation_radius
    # quadratic penalty near obstacle (tune factor)
    penalty[inside] = ((inflation_radius - capped[inside]) ** 2) * 8.0
    # apply penalty (additive)
    weight += penalty
    # normalize & rescale to keep costs in a reasonable range (preserve very large for blocked)
    finite_mask = np.isfinite(weight) & (weight < 1e5)
    if finite_mask.any():
        wmin = weight[finite_mask].min()
        wmax = weight[finite_mask].max()
        # scale to [1, 200] range (avoid shrinking very large blocked weights)
        if wmax - wmin > 1e-6:
            rescaled = 1.0 + (weight - wmin) / (wmax - wmin) * 199.0
            weight[finite_mask] = rescaled[finite_mask]

    # ensure blocked cells remain extremely costly
    weight[height > 1.6] = 1e6

    return weight.astype(np.float32)
