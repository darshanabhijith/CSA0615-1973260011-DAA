"""CSA0615 - Design and Analysis of Algorithms, Assignment 4B.

All six tasks are implemented with small, reproducible demonstration inputs.
Run with Python 3.10+ using:
    python CSA0615_DAA_Assignment_4B_Darshan_A.py
"""

from __future__ import annotations

from collections import defaultdict
from heapq import heappop, heappush
from itertools import permutations
from math import inf
from time import perf_counter
import tracemalloc


# ---------------------------------------------------------------------------
# TASK 1: WORD WRAP USING DYNAMIC PROGRAMMING
# ---------------------------------------------------------------------------

def word_wrap(words: list[str], width: int) -> dict:
    """Return the minimum-square-slack layout for a list of words."""
    if width <= 0:
        raise ValueError("Line width must be positive.")
    if any(len(word) > width for word in words):
        raise ValueError("A word is longer than the selected line width.")

    n = len(words)
    dp = [inf] * (n + 1)
    previous = [-1] * (n + 1)
    dp[0] = 0

    for end in range(1, n + 1):
        line_length = 0
        for start in range(end - 1, -1, -1):
            line_length += len(words[start])
            if start < end - 1:
                line_length += 1
            if line_length > width:
                break
            unused = width - line_length
            line_penalty = 0 if end == n else unused ** 2
            candidate = dp[start] + line_penalty
            if candidate < dp[end]:
                dp[end] = candidate
                previous[end] = start

    if previous[n] == -1 and n > 0:
        raise ValueError("No valid word arrangement exists.")

    lines = []
    end = n
    while end > 0:
        start = previous[end]
        line_words = words[start:end]
        used = sum(map(len, line_words)) + len(line_words) - 1
        unused = width - used
        penalty = 0 if end == n else unused ** 2
        lines.append({"words": line_words, "unused": unused, "penalty": penalty})
        end = start
    lines.reverse()
    return {"minimum_penalty": dp[n], "lines": lines, "dp": dp, "previous": previous}


def run_task1() -> None:
    words = ["Dynamic", "programming", "helps", "arrange", "words", "into",
             "clear", "document", "lines", "with", "minimum", "unused",
             "space", "penalty"]
    print("\n========== TASK 1: WORD WRAP ==========")
    for width in (20, 24, 30):
        result = word_wrap(words, width)
        print(f"\nWidth = {width}; minimum penalty = {result['minimum_penalty']}")
        print("DP prefix costs:", result["dp"])
        for number, line in enumerate(result["lines"], 1):
            print(f"Line {number}: {' '.join(line['words'])} | unused={line['unused']} | penalty={line['penalty']}")

    boundary = word_wrap(["data", "maps"], 9)
    print("\nBoundary case (width 9, exact fit):", boundary)
    try:
        word_wrap(["algorithm"], 5)
    except ValueError as error:
        print("Overlong-word boundary check:", error)


# ---------------------------------------------------------------------------
# TASK 2: FRACTIONAL GREEDY AND 0/1 DYNAMIC PROGRAMMING KNAPSACK
# ---------------------------------------------------------------------------

def fractional_knapsack(items: list[tuple[str, int, int]], capacity: int) -> dict:
    """Greedy fractional knapsack. Each tuple is (id, weight, value)."""
    if capacity < 0 or any(w <= 0 or v < 0 for _, w, v in items):
        raise ValueError("Capacity must be non-negative; weights positive and values non-negative.")
    ordered = sorted(items, key=lambda item: (-item[2] / item[1], item[0]))
    remaining = capacity
    selected = []
    total_value = 0.0
    for item_id, weight, value in ordered:
        if remaining == 0:
            break
        fraction = min(1.0, remaining / weight)
        selected.append((item_id, fraction, fraction * weight, fraction * value))
        remaining -= fraction * weight
        total_value += fraction * value
    return {"selected": selected, "value": total_value, "used_weight": capacity - remaining,
            "order": [(item_id, value / weight) for item_id, weight, value in ordered]}


def zero_one_knapsack(items: list[tuple[str, int, int]], capacity: int) -> dict:
    """0/1 knapsack DP with selected-item reconstruction and a cell counter."""
    if capacity < 0 or any(w <= 0 or v < 0 for _, w, v in items):
        raise ValueError("Capacity must be non-negative; weights positive and values non-negative.")
    n = len(items)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]
    updates = 0
    for i, (_, weight, value) in enumerate(items, 1):
        for cap in range(capacity + 1):
            dp[i][cap] = dp[i - 1][cap]
            if weight <= cap:
                include = dp[i - 1][cap - weight] + value
                if include > dp[i][cap]:
                    dp[i][cap] = include
                updates += 1
    selected = []
    cap = capacity
    for i in range(n, 0, -1):
        if dp[i][cap] != dp[i - 1][cap]:
            item_id, weight, value = items[i - 1]
            selected.append((item_id, weight, value))
            cap -= weight
    selected.reverse()
    return {"value": dp[n][capacity], "selected": selected, "used_weight": sum(x[1] for x in selected),
            "table": dp, "cells": n * (capacity + 1), "include_checks": updates}


def run_task2() -> None:
    print("\n========== TASK 2: GREEDY vs DYNAMIC PROGRAMMING ==========")
    normal_items = [("A", 10, 60), ("B", 20, 100), ("C", 30, 120), ("D", 5, 30)]
    capacity = 50
    fractional = fractional_knapsack(normal_items, capacity)
    exact = zero_one_knapsack(normal_items, capacity)
    print("Normal dataset:", normal_items, "capacity:", capacity)
    print("Greedy ratio order:", fractional["order"])
    print("Fractional picks (id, fraction, weight, value):", fractional["selected"])
    print("Fractional value:", round(fractional["value"], 2))
    print("0/1 DP selected:", exact["selected"], "weight:", exact["used_weight"], "value:", exact["value"])

    counterexample = [("A", 10, 60), ("B", 20, 100), ("C", 30, 120)]
    greedy_whole = fractional_knapsack(counterexample, 50)
    dp_result = zero_one_knapsack(counterexample, 50)
    print("\n0/1 counterexample:", counterexample, "capacity: 50")
    print("Ratio-greedy first commitment:", greedy_whole["selected"][0])
    # A whole-item version of the ratio heuristic is shown explicitly.
    remaining = 50
    greedy_value = 0
    greedy_items = []
    for item_id, ratio in greedy_whole["order"]:
        item = next(item for item in counterexample if item[0] == item_id)
        if item[1] <= remaining:
            greedy_items.append(item[0])
            greedy_value += item[2]
            remaining -= item[1]
    print("Whole-item ratio greedy selected:", greedy_items, "value:", greedy_value, "unused capacity:", remaining)
    print("0/1 DP selected:", dp_result["selected"], "value:", dp_result["value"])
    print("DP table cells:", dp_result["cells"], "include checks:", dp_result["include_checks"])


# ---------------------------------------------------------------------------
# TASK 3: SHORTEST-PATH ROUTING
# ---------------------------------------------------------------------------

VERTICES = list("ABCDEFGH")
BASE_EDGES = [
    ("A", "B", 2), ("A", "C", 5), ("B", "D", 2), ("C", "B", 7),
    ("C", "E", 2), ("D", "E", 3), ("D", "F", 4), ("E", "F", 1),
    ("E", "G", 5), ("F", "G", 2), ("G", "H", 2), ("F", "H", 5),
    ("B", "H", 12), ("D", "G", 7),
]


def dijkstra(vertices: list[str], edges: list[tuple[str, str, int]], source: str) -> dict:
    adjacency = defaultdict(list)
    for u, v, w in edges:
        adjacency[u].append((v, w))
    distance = {v: inf for v in vertices}
    predecessor = {v: None for v in vertices}
    distance[source] = 0
    heap = [(0, source)]
    finalised = []
    done = set()
    while heap:
        current_distance, u = heappop(heap)
        if u in done or current_distance != distance[u]:
            continue
        done.add(u)
        finalised.append(u)
        for v, weight in adjacency[u]:
            if v not in done and current_distance + weight < distance[v]:
                distance[v] = current_distance + weight
                predecessor[v] = u
                heappush(heap, (distance[v], v))
    return {"distance": distance, "predecessor": predecessor, "finalised": finalised}


def bellman_ford(vertices: list[str], edges: list[tuple[str, str, int]], source: str) -> dict:
    distance = {v: inf for v in vertices}
    predecessor = {v: None for v in vertices}
    distance[source] = 0
    rounds = []
    for iteration in range(1, len(vertices)):
        changed = False
        for u, v, weight in edges:
            if distance[u] != inf and distance[u] + weight < distance[v]:
                distance[v] = distance[u] + weight
                predecessor[v] = u
                changed = True
        rounds.append((iteration, dict(distance)))
        if not changed:
            break
    negative_cycle = any(distance[u] != inf and distance[u] + weight < distance[v]
                         for u, v, weight in edges)
    return {"distance": distance, "predecessor": predecessor, "rounds": rounds,
            "negative_cycle": negative_cycle}


def reconstruct_path(predecessor: dict, target: str) -> list[str]:
    if target not in predecessor:
        return []
    path = []
    current = target
    seen = set()
    while current is not None:
        if current in seen:
            return []
        seen.add(current)
        path.append(current)
        current = predecessor[current]
    return list(reversed(path))


def floyd_warshall(vertices: list[str], edges: list[tuple[str, str, int]]) -> list[list[float]]:
    index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    distance = [[inf] * n for _ in range(n)]
    for i in range(n):
        distance[i][i] = 0
    for u, v, weight in edges:
        distance[index[u]][index[v]] = min(distance[index[u]][index[v]], weight)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if distance[i][k] + distance[k][j] < distance[i][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
    return distance


def warshall_reachability(vertices: list[str], edges: list[tuple[str, str, int]]) -> list[list[int]]:
    index = {v: i for i, v in enumerate(vertices)}
    n = len(vertices)
    reach = [[0] * n for _ in range(n)]
    for i in range(n):
        reach[i][i] = 1
    for u, v, _ in edges:
        reach[index[u]][index[v]] = 1
    for k in range(n):
        for i in range(n):
            for j in range(n):
                reach[i][j] = int(bool(reach[i][j] or (reach[i][k] and reach[k][j])))
    return reach


def print_matrix(vertices: list[str], matrix: list[list], infinity_text: str = "INF") -> None:
    print("      " + " ".join(f"{v:>5}" for v in vertices))
    for v, row in zip(vertices, matrix):
        values = [infinity_text if value == inf else str(int(value)) for value in row]
        print(f"{v:>5} " + " ".join(f"{value:>5}" for value in values))


def run_task3() -> None:
    print("\n========== TASK 3: ROUTING ENGINE ==========")
    dijkstra_result = dijkstra(VERTICES, BASE_EDGES, "A")
    print("Base graph edges:", len(BASE_EDGES))
    print("Dijkstra finalisation order:", dijkstra_result["finalised"])
    for vertex in VERTICES:
        print(vertex, "distance=", dijkstra_result["distance"][vertex],
              "predecessor=", dijkstra_result["predecessor"][vertex],
              "path=", reconstruct_path(dijkstra_result["predecessor"], vertex))

    negative_edges = [(u, v, -6 if (u, v) == ("C", "B") else w) for u, v, w in BASE_EDGES]
    bad_dijkstra = dijkstra(VERTICES, negative_edges, "A")
    bf_result = bellman_ford(VERTICES, negative_edges, "A")
    print("\nChanged edge C->B from +7 to -6 (no negative cycle).")
    print("Prediction: Dijkstra's finalised B will not be reopened; Bellman-Ford should improve B through C.")
    print("Dijkstra B distance:", bad_dijkstra["distance"]["B"], "(incorrect for negative-edge graph)")
    print("Bellman-Ford B distance:", bf_result["distance"]["B"], "path:", reconstruct_path(bf_result["predecessor"], "B"))
    print("Bellman-Ford distances:", bf_result["distance"])

    negative_cycle_edges = negative_edges + [("H", "C", -10)]
    cycle_result = bellman_ford(VERTICES, negative_cycle_edges, "A")
    print("\nAdded H->C = -10. Reachable negative cycle detected:", cycle_result["negative_cycle"])

    print("\nFloyd all-pairs shortest-distance matrix on the original non-negative graph:")
    print_matrix(VERTICES, floyd_warshall(VERTICES, BASE_EDGES))
    print("\nWarshall reachability matrix (1 = reachable, 0 = not reachable):")
    print_matrix(VERTICES, warshall_reachability(VERTICES, BASE_EDGES), infinity_text="-")


# ---------------------------------------------------------------------------
# TASK 4: TWO-LINE ASSEMBLY-LINE SCHEDULING
# ---------------------------------------------------------------------------

def assembly_line_schedule(a1: list[int], a2: list[int], t1: list[int], t2: list[int],
                           entry1: int, entry2: int, exit1: int, exit2: int) -> dict:
    if len(a1) != len(a2) or not a1:
        raise ValueError("Both assembly lines must have the same positive number of stations.")
    n = len(a1)
    if len(t1) != n - 1 or len(t2) != n - 1:
        raise ValueError("Transfer arrays must have one value between each pair of stations.")
    f1 = [0] * n
    f2 = [0] * n
    parent1 = [0] * n  # line from which the optimal route reached line 1
    parent2 = [0] * n  # line from which the optimal route reached line 2
    f1[0] = entry1 + a1[0]
    f2[0] = entry2 + a2[0]
    trace = []
    for i in range(1, n):
        stay1 = f1[i - 1] + a1[i]
        transfer1 = f2[i - 1] + t2[i - 1] + a1[i]
        if stay1 <= transfer1:
            f1[i], parent1[i], choice1 = stay1, 1, "stay on L1"
        else:
            f1[i], parent1[i], choice1 = transfer1, 2, "transfer L2->L1"

        stay2 = f2[i - 1] + a2[i]
        transfer2 = f1[i - 1] + t1[i - 1] + a2[i]
        if stay2 <= transfer2:
            f2[i], parent2[i], choice2 = stay2, 2, "stay on L2"
        else:
            f2[i], parent2[i], choice2 = transfer2, 1, "transfer L1->L2"
        trace.append({"station": i + 1, "stay1": stay1, "transfer1": transfer1,
                      "chosen1": choice1, "f1": f1[i], "stay2": stay2,
                      "transfer2": transfer2, "chosen2": choice2, "f2": f2[i]})

    total1 = f1[-1] + exit1
    total2 = f2[-1] + exit2
    final_line = 1 if total1 <= total2 else 2
    minimum = min(total1, total2)
    line_at_station = [0] * n
    current = final_line
    for i in range(n - 1, -1, -1):
        line_at_station[i] = current
        if i > 0:
            current = parent1[i] if current == 1 else parent2[i]
    transfers = [i + 1 for i in range(1, n) if line_at_station[i] != line_at_station[i - 1]]
    return {"minimum_time": minimum, "line_at_station": line_at_station,
            "transfers_before_station": transfers, "trace": trace,
            "f1": f1, "f2": f2, "exit_totals": (total1, total2)}


def run_task4() -> None:
    print("\n========== TASK 4: ASSEMBLY-LINE SCHEDULING ==========")
    a1 = [7, 9, 3, 4, 8, 4]
    a2 = [8, 5, 6, 4, 5, 7]
    t1 = [2, 3, 1, 3, 4]  # L1 -> L2
    t2 = [2, 1, 2, 2, 1]  # L2 -> L1
    args = (a1, a2, t1, t2, 2, 4, 3, 2)
    result = assembly_line_schedule(*args)
    print("Line 1 station times:", a1, "Line 2:", a2)
    print("Transfer L1->L2:", t1, "Transfer L2->L1:", t2)
    for row in result["trace"]:
        print(row)
    print("DP costs L1:", result["f1"], "L2:", result["f2"])
    print("Route:", " -> ".join(f"L{line}-S{station}" for station, line in enumerate(result["line_at_station"], 1)))
    print("Transfer before station(s):", result["transfers_before_station"])
    print("Exit totals:", result["exit_totals"], "minimum time:", result["minimum_time"])

    changed_t1 = t1.copy()
    changed_t1[2] = 20
    changed_transfer = assembly_line_schedule(a1, a2, changed_t1, t2, 2, 4, 3, 2)
    print("\nPrediction: increasing L1->L2 transfer before station 4 from 1 to 20 may discourage a switch there; total time should not decrease.")
    print("Observed route after transfer change:", changed_transfer["line_at_station"],
          "minimum:", changed_transfer["minimum_time"])

    changed_a1 = a1.copy()
    changed_a1[2] = 30
    disruption = assembly_line_schedule(changed_a1, a2, t1, t2, 2, 4, 3, 2)
    print("\nPrediction: making Line 1 station 3 much slower should shift the route to Line 2 at or before station 3.")
    print("Observed route after station disruption:", disruption["line_at_station"],
          "minimum:", disruption["minimum_time"])


# ---------------------------------------------------------------------------
# TASK 5: MINIMUM SPANNING TREE - KRUSKAL, PRIM, BORUVKA
# ---------------------------------------------------------------------------

MST_VERTICES = list("ABCDEFGH")
MST_EDGES = [
    ("A", "B", 4), ("A", "C", 3), ("B", "C", 1), ("B", "D", 2),
    ("C", "D", 4), ("C", "E", 2), ("D", "E", 3), ("D", "F", 7),
    ("E", "F", 2), ("E", "G", 5), ("F", "G", 3), ("F", "H", 6),
    ("G", "H", 2), ("D", "H", 9), ("E", "H", 8),
]


class DisjointSet:
    def __init__(self, values: list[str]):
        self.parent = {x: x for x in values}
        self.rank = {x: 0 for x in values}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str) -> bool:
        root_a, root_b = self.find(a), self.find(b)
        if root_a == root_b:
            return False
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1
        return True


def kruskal(vertices: list[str], edges: list[tuple[str, str, int]]) -> dict:
    dsu = DisjointSet(vertices)
    chosen, trace = [], []
    for u, v, w in sorted(edges, key=lambda edge: (edge[2], edge[0], edge[1])):
        accepted = dsu.union(u, v)
        trace.append((u, v, w, "accept" if accepted else "reject-cycle"))
        if accepted:
            chosen.append((u, v, w))
        if len(chosen) == len(vertices) - 1:
            break
    return {"edges": chosen, "cost": sum(e[2] for e in chosen), "trace": trace}


def prim(vertices: list[str], edges: list[tuple[str, str, int]], start: str = "A",
         prefer_high_tie: bool = False) -> dict:
    adjacency = defaultdict(list)
    for u, v, w in edges:
        adjacency[u].append((v, w))
        adjacency[v].append((u, w))
    visited = {start}
    heap = []

    def add_edges(u: str) -> None:
        for v, w in adjacency[u]:
            if v not in visited:
                tie_key = -ord(v[0]) if prefer_high_tie else ord(v[0])
                source_tie_key = -ord(u[0]) if prefer_high_tie else ord(u[0])
                heappush(heap, (w, tie_key, source_tie_key, u, v))

    add_edges(start)
    chosen, trace = [], []
    while heap and len(chosen) < len(vertices) - 1:
        weight, _, _, u, v = heappop(heap)
        if v in visited:
            trace.append((u, v, weight, "reject-already-connected"))
            continue
        visited.add(v)
        chosen.append((u, v, weight))
        trace.append((u, v, weight, "accept"))
        add_edges(v)
    return {"edges": chosen, "cost": sum(e[2] for e in chosen), "trace": trace}


def boruvka(vertices: list[str], edges: list[tuple[str, str, int]]) -> dict:
    dsu = DisjointSet(vertices)
    chosen, rounds = [], []
    while len(chosen) < len(vertices) - 1:
        cheapest = {}
        for u, v, w in edges:
            root_u, root_v = dsu.find(u), dsu.find(v)
            if root_u == root_v:
                continue
            candidate = (w, u, v)
            if root_u not in cheapest or candidate < cheapest[root_u]:
                cheapest[root_u] = candidate
            if root_v not in cheapest or candidate < cheapest[root_v]:
                cheapest[root_v] = candidate
        round_edges = []
        merged = 0
        for _, u, v in sorted(set(cheapest.values())):
            w = next(weight for x, y, weight in edges if (x, y) == (u, v))
            if dsu.union(u, v):
                chosen.append((u, v, w))
                round_edges.append((u, v, w))
                merged += 1
        rounds.append(round_edges)
        if merged == 0:
            break
    return {"edges": chosen, "cost": sum(e[2] for e in chosen), "rounds": rounds}


def run_task5() -> None:
    print("\n========== TASK 5: MINIMUM SPANNING TREE ==========")
    for name, result in (("Kruskal", kruskal(MST_VERTICES, MST_EDGES)),
                         ("Prim", prim(MST_VERTICES, MST_EDGES)),
                         ("Boruvka", boruvka(MST_VERTICES, MST_EDGES))):
        print(name, "trace:", result.get("trace", result.get("rounds")))
        print(name, "MST edges:", result["edges"], "total:", result["cost"])

    tied = [
        ("A", "B", 1), ("A", "C", 1), ("B", "C", 1), ("B", "D", 9),
        ("C", "D", 1), ("C", "E", 10), ("D", "E", 2), ("D", "F", 8),
        ("E", "F", 2), ("E", "G", 9), ("F", "G", 2), ("F", "H", 8),
        ("G", "H", 2), ("D", "H", 10), ("E", "H", 10),
    ]
    k = kruskal(MST_VERTICES, tied)
    p = prim(MST_VERTICES, tied, prefer_high_tie=True)
    b = boruvka(MST_VERTICES, tied)
    print("\nTie-stressed graph:")
    print("Kruskal:", k["edges"], "cost:", k["cost"])
    print("Prim (high-neighbour tie rule):", p["edges"], "cost:", p["cost"])
    print("Boruvka:", b["edges"], "cost:", b["cost"])


# ---------------------------------------------------------------------------
# TASK 6: TRAVELLING SALESPERSON - HELD-KARP DYNAMIC PROGRAMMING
# ---------------------------------------------------------------------------

TSP_LABELS = ["Hub", "A", "B", "C", "D", "E", "F"]
TSP_MATRIX = [
    [0, 10, 15, 20, 10, 25, 18],
    [10, 0, 9, 10, 12, 17, 20],
    [15, 9, 0, 8, 10, 14, 16],
    [20, 10, 8, 0, 11, 13, 15],
    [10, 12, 10, 11, 0, 9, 12],
    [25, 17, 14, 13, 9, 0, 7],
    [18, 20, 16, 15, 12, 7, 0],
]


def held_karp(matrix: list[list[int]], labels: list[str]) -> dict:
    """Exact TSP DP. Matrix index 0 is the hub; remaining vertices are deliveries."""
    total_vertices = len(matrix)
    if total_vertices != len(labels) or total_vertices < 2:
        raise ValueError("A square cost matrix and matching labels are required.")
    n = total_vertices - 1
    if any(len(row) != total_vertices for row in matrix):
        raise ValueError("Travel cost matrix must be square.")

    dp = {}
    parent = {}
    evaluated = 0
    for j in range(n):
        mask = 1 << j
        dp[(mask, j)] = matrix[0][j + 1]
        parent[(mask, j)] = None
        evaluated += 1

    for mask in range(1, 1 << n):
        for j in range(n):
            if not (mask & (1 << j)) or mask == (1 << j):
                continue
            previous_mask = mask ^ (1 << j)
            best_cost = inf
            best_previous = None
            for k in range(n):
                if previous_mask & (1 << k):
                    candidate = dp[(previous_mask, k)] + matrix[k + 1][j + 1]
                    if candidate < best_cost:
                        best_cost, best_previous = candidate, k
            dp[(mask, j)] = best_cost
            parent[(mask, j)] = best_previous
            evaluated += 1

    full = (1 << n) - 1
    best_tour_cost = inf
    last = None
    for j in range(n):
        candidate = dp[(full, j)] + matrix[j + 1][0]
        if candidate < best_tour_cost:
            best_tour_cost, last = candidate, j

    reverse_route = []
    mask = full
    current = last
    while current is not None:
        reverse_route.append(current + 1)
        previous = parent[(mask, current)]
        mask ^= 1 << current
        current = previous
    route_indices = [0] + list(reversed(reverse_route)) + [0]
    route = [labels[i] for i in route_indices]
    return {"cost": best_tour_cost, "route": route, "route_indices": route_indices,
            "states": evaluated, "dp": dp, "parent": parent}


def exhaustive_tsp(matrix: list[list[int]]) -> tuple[int, tuple[int, ...]]:
    n = len(matrix) - 1
    best_cost = inf
    best_route = None
    for order in permutations(range(1, n + 1)):
        route = (0,) + order + (0,)
        cost = sum(matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
        if cost < best_cost:
            best_cost, best_route = cost, route
    return best_cost, best_route


def run_task6() -> None:
    print("\n========== TASK 6: TSP USING DYNAMIC PROGRAMMING ==========")
    print("Cost matrix labels:", TSP_LABELS)
    for row in TSP_MATRIX:
        print(row)
    result = held_karp(TSP_MATRIX, TSP_LABELS)
    print("Optimal route:", " -> ".join(result["route"]))
    print("Minimum total travel cost:", result["cost"])
    print("DP states evaluated:", result["states"])

    # Validate a 4-delivery-location instance using independent exhaustive search.
    small_labels = ["Hub", "A", "B", "C", "D"]
    small_matrix = [row[:5] for row in TSP_MATRIX[:5]]
    dp_small = held_karp(small_matrix, small_labels)
    brute_cost, brute_route = exhaustive_tsp(small_matrix)
    print("\nValidation with 4 delivery locations:")
    print("DP route:", dp_small["route"], "cost:", dp_small["cost"])
    print("Exhaustive optimum indices:", brute_route, "cost:", brute_cost)
    print("Validation matches:", dp_small["cost"] == brute_cost)

    print("\nState-space measurements (times and memory depend on the computer running the program):")
    for n in range(4, 9):
        labels = ["Hub"] + [f"L{i}" for i in range(1, n + 1)]
        # Deterministic complete symmetric test matrix.
        matrix = [[0 if i == j else 4 + ((i * 7 + j * 11 + i * j) % 23)
                   for j in range(n + 1)] for i in range(n + 1)]
        tracemalloc.start()
        start = perf_counter()
        measurement = held_karp(matrix, labels)
        elapsed = perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        print(f"locations={n}, states={measurement['states']}, expected={n * (2 ** (n - 1))}, "
              f"time={elapsed:.6f}s, peak_memory={peak / 1024:.1f} KiB")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_task1()
    run_task2()
    run_task3()
    run_task4()
    run_task5()
    run_task6()
