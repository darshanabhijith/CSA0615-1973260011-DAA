"""CSA06 - Design and Analysis of Algorithms, Assignment 5 (CO5)
Student: Darshan A | Register No.: 1973260011

The program contains the implementations and small experiments used in the report.
It uses only Python's standard library.
"""

from itertools import combinations
import random


# ------------------------------------------------------------
# B1 and B2: Graph colouring with backtracking and counters
# ------------------------------------------------------------

def graph_colouring(vertex_count, edges, colour_count=3, use_degree_order=False):
    """Return (success, colouring, statistics, order).

    Statistics:
      nodes       = recursive states entered, including the root
      attempts    = colours checked
      rejected    = colour choices rejected by the neighbour test
      backtracks  = successful assignments later undone after failure
      max_depth   = maximum number of assigned vertices
    """
    adjacency = [set() for _ in range(vertex_count)]
    for u, v in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)

    if use_degree_order:
        # Highest-degree vertices are selected first. Ties use vertex number.
        order = sorted(range(vertex_count),
                       key=lambda vertex: (-len(adjacency[vertex]), vertex))
    else:
        order = list(range(vertex_count))

    assigned = {}
    stats = {
        "nodes": 0,
        "attempts": 0,
        "rejected": 0,
        "backtracks": 0,
        "max_depth": 0,
    }

    def search(position):
        stats["nodes"] += 1
        stats["max_depth"] = max(stats["max_depth"], position)

        if position == vertex_count:
            return True

        vertex = order[position]
        for colour in range(colour_count):
            stats["attempts"] += 1
            conflict = False
            for neighbour in adjacency[vertex]:
                if neighbour in assigned and assigned[neighbour] == colour:
                    conflict = True
                    break

            if conflict:
                stats["rejected"] += 1
                continue

            assigned[vertex] = colour
            if search(position + 1):
                return True

            # Undo the choice before trying the next colour.
            del assigned[vertex]
            stats["backtracks"] += 1

        return False

    success = search(0)
    colouring = dict(assigned) if success else None
    return success, colouring, stats, order


def graph_test_suite():
    """Three tests: colourable, not colourable with 3 colours, and boundary."""
    path_edges = [(i, i + 1) for i in range(7)]
    # V1-V4 are isolated; V5-V8 form a complete graph K4.
    hard_edges = [(i, j) for i in range(4, 8) for j in range(i + 1, 8)]
    boundary_edges = []

    tests = [
        ("Path with 8 vertices", 8, path_edges),
        ("Four isolated vertices followed by K4", 8, hard_edges),
        ("One isolated vertex (boundary)", 1, boundary_edges),
    ]

    for name, count, edges in tests:
        print("\n", name)
        for label, heuristic in (("Input order", False), ("Degree order", True)):
            success, colouring, stats, order = graph_colouring(
                count, edges, colour_count=3, use_degree_order=heuristic
            )
            print(label, "success=", success, "colouring=", colouring,
                  "order=", order, "stats=", stats)


# ------------------------------------------------------------
# B3: Exact and greedy Set Cover
# ------------------------------------------------------------

def exact_set_cover(universe, centres):
    """Try centre combinations in increasing size; return optimum and checks."""
    names = list(centres.keys())
    required = set(universe)
    combinations_checked = 0

    for size in range(1, len(names) + 1):
        for chosen in combinations(names, size):
            combinations_checked += 1
            covered = set()
            for centre in chosen:
                covered.update(centres[centre])
            if covered == required:
                return list(chosen), combinations_checked

    return None, combinations_checked


def greedy_set_cover(universe, centres):
    """Repeatedly choose the centre covering most uncovered zones.

    Ties are settled by the order in which centres appear in the dictionary.
    Return selected centres, scoring checks, and a step-by-step trace.
    """
    uncovered = set(universe)
    selected = []
    scoring_checks = 0
    trace = []

    while uncovered:
        best_centre = None
        best_new_zones = set()

        for centre, zones in centres.items():
            if centre in selected:
                continue
            scoring_checks += 1
            newly_covered = uncovered.intersection(zones)
            if len(newly_covered) > len(best_new_zones):
                best_centre = centre
                best_new_zones = newly_covered

        if not best_new_zones:
            return None, scoring_checks, trace

        selected.append(best_centre)
        trace.append((best_centre, sorted(best_new_zones), len(best_new_zones)))
        uncovered.difference_update(best_new_zones)

    return selected, scoring_checks, trace


def make_scale_instance(zone_count, centre_count, seed):
    """Create a repeatable small test instance using a fixed random seed."""
    generator = random.Random(seed)
    zones = ["Z" + str(i) for i in range(1, zone_count + 1)]
    coverage_size = max(2, round(zone_count * 0.32))
    centres = {}

    for index in range(centre_count):
        name = "C" + str(index + 1)
        centres[name] = set(generator.sample(zones, coverage_size))

    # Make sure every zone appears in at least one centre.
    all_covered = set().union(*centres.values())
    for zone in sorted(set(zones) - all_covered):
        centres["C1"].add(zone)

    return zones, centres


def set_cover_tests():
    cases = [
        ("Instance 1", set("1234"), {
            "A": set("123"), "B": set("4"),
            "C": set("12"), "D": set("34")
        }),
        ("Instance 2", set("12345"), {
            "A": set("123"), "B": set("45"),
            "C": set("14"), "D": set("25"), "E": set("34")
        }),
        ("Counterexample", set("123456"), {
            "A": set("1234"), "B": set("125"),
            "C": set("346"), "D": set("5"), "E": set("6")
        }),
    ]

    for name, universe, centres in cases:
        exact, exact_checks = exact_set_cover(universe, centres)
        greedy, greedy_checks, trace = greedy_set_cover(universe, centres)
        print("\n", name)
        print("Exact:", exact, "size:", len(exact), "combinations:", exact_checks)
        print("Greedy:", greedy, "size:", len(greedy), "centre checks:", greedy_checks)
        print("Greedy trace:", trace)

    print("\nScale tests (fixed seeds)")
    for zone_count, centre_count, seed in ((8, 10, 51), (12, 16, 52), (16, 20, 53)):
        zones, centres = make_scale_instance(zone_count, centre_count, seed)
        exact, exact_checks = exact_set_cover(zones, centres)
        greedy, greedy_checks, trace = greedy_set_cover(zones, centres)
        print(zone_count, "zones,", centre_count, "centres:",
              "exact size", len(exact), "exact checks", exact_checks,
              "greedy size", len(greedy), "greedy checks", greedy_checks)


# ------------------------------------------------------------
# B4: First Fit Decreasing for Bin Packing + small exact check
# ------------------------------------------------------------

def first_fit_decreasing(workloads, capacity):
    """Place larger workloads first into the first server with enough space."""
    ordered = sorted(enumerate(workloads, start=1),
                     key=lambda item: (-item[1], item[0]))
    servers = []

    for workload_id, demand in ordered:
        placed = False
        for server in servers:
            used = sum(weight for _, weight in server)
            if used + demand <= capacity:
                server.append((workload_id, demand))
                placed = True
                break
        if not placed:
            servers.append([(workload_id, demand)])

    return servers


def exact_bin_packing(workloads, capacity):
    """Small exact branch-and-bound solver used only for the counterexample."""
    ordered = sorted(workloads, reverse=True)
    best_count = len(ordered)
    best_servers = None
    search_nodes = 0
    servers = []

    def search(index):
        nonlocal best_count, best_servers, search_nodes
        search_nodes += 1

        if index == len(ordered):
            if len(servers) < best_count:
                best_count = len(servers)
                best_servers = [server[:] for server in servers]
            return

        if len(servers) >= best_count:
            return

        demand = ordered[index]
        tried_remaining = set()
        for server in servers:
            remaining = capacity - sum(server)
            if remaining >= demand and remaining not in tried_remaining:
                tried_remaining.add(remaining)
                server.append(demand)
                search(index + 1)
                server.pop()

        if len(servers) + 1 < best_count:
            servers.append([demand])
            search(index + 1)
            servers.pop()

    search(0)
    return best_count, best_servers, search_nodes


def bin_packing_tests():
    capacity = 10
    workloads = [8, 7, 6, 5, 4, 3, 2, 2]
    servers = first_fit_decreasing(workloads, capacity)
    print("Production example, capacity", capacity)
    print("Number of servers:", len(servers))
    for index, server in enumerate(servers, start=1):
        used = sum(weight for _, weight in server)
        print("Server", index, server, "remaining capacity", capacity - used)

    challenging = [2, 2, 2, 3, 5, 6]
    heuristic_servers = first_fit_decreasing(challenging, capacity)
    optimum, exact_servers, exact_nodes = exact_bin_packing(challenging, capacity)
    print("\nChallenge input:", challenging)
    print("FFD servers:", heuristic_servers)
    print("FFD server count:", len(heuristic_servers))
    print("Exact minimum:", optimum, "one optimum packing:", exact_servers,
          "exact search nodes:", exact_nodes)


if __name__ == "__main__":
    print("CSA06 CO5 - Assignment 5 experiments")
    graph_test_suite()
    set_cover_tests()
    bin_packing_tests()
