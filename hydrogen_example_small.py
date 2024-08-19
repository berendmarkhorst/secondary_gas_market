from objects import *
import networkx as nx
import pandas as pd

# Sets
nodes = [1, 2, 3]
arcs = [(1, 2), (2, 3), (2, 1), (3, 2)]
digraph = nx.DiGraph()
digraph.add_nodes_from(nodes)
digraph.add_edges_from(arcs)

traders = [1]
commodities = ["gas", "hydrogen"]

# Parameters
loss_rate = 0.2
allowed_percentage = {node: 1 for node in nodes}
probability4 = 1
gamma = 0.25

# Arc costs and capacities
arc_costs = {(arc, k): 10 for arc in arcs for k in commodities}
arc_capacities = {arc: 30 for arc in arcs}

# Entry and exit costs
entry_costs3 = {(node, k): 2 for node in nodes for k in commodities}
exit_costs3 = {(node, k): 2 for node in nodes for k in commodities}

# Node capacities and demands
node_capacities = {(node, k): 30 for node in nodes for k in commodities}
node_demands3yes = {(1, "gas"): 0, (2, "gas"): 0, (3, "gas"): 3.2,
                 (1, "hydrogen"): 0, (2, "hydrogen"): 1, (3, "hydrogen"): 0}
node_demands3no = {(1, "gas"): 0, (2, "gas"): 4, (3, "gas"): 0,
                 (1, "hydrogen"): 0, (2, "hydrogen"): 2, (3, "hydrogen"): 0}

# Production costs and capacities
production_costs3 = {(1, 1, "gas"): 10, (2, 1, "gas"): 50, (3, 1, "gas"): 20,
                     (1, 1, "hydrogen"): 10, (2, 1, "hydrogen"): 50, (3, 1, "hydrogen"): 20,
                     }
production_capacities3 = {(1, 1, "gas"): 10, (1, 2, "gas"): 0, (1, 3, "gas"): 0,
                          (1, 1, "hydrogen"): 30, (1, 2, "hydrogen"): 0, (1, 3, "hydrogen"): 30,
                          }

# Storage costs and capacities
storage_costs3 = {(1, 1, "gas"): 2, (1, 2, "gas"): 2, (1, 3, "gas"): 2,
                  (1, 1, "hydrogen"): 2, (1, 2, "hydrogen"): 2, (1, 3, "hydrogen"): 2,
                  }
storage_capacities = {(1, 1, "gas"): 100, (1, 2, "gas"): 100, (1, 3, "gas"): 100,
                      (1, 1, "hydrogen"): 100, (1, 2, "hydrogen"): 100, (1, 3, "hydrogen"): 100,
                      }

# Third stage: dummy for now
stage4 = Stage(1, "intra day", arc_costs, entry_costs3, exit_costs3, arc_capacities, node_capacities, probability4, node_demands3no, production_costs3, production_capacities3, storage_costs3, storage_capacities, None)
# stage5 = Stage(2, "intra day", arc_costs, entry_costs3, exit_costs3, arc_capacities, node_capacities, probability4, node_demands3no, production_costs3, production_capacities3, storage_costs3, storage_capacities, stage4)

# Stages objects
stages = [stage4]

# Problem object
problem = Problem(digraph, stages, traders, loss_rate, allowed_percentage, commodities, gamma)

model = problem.build_model()
model.optimize()

print("Flow variables")
for t in problem.traders:
    for a in problem.arcs:
        for m in problem.third_stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'f[{t},{a[0]},{a[1]},{m},{k}]').X
                if value > 0.1:
                    print(f"Trader {t}, Arc {a}, Stage {m}, Commodity {k}: {value}")

print("\nStorage variables")
for t in problem.traders:
    for n in problem.nodes:
        for m in problem.third_stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'v[{t},{n},{m},{k}]').X
                if value > 0:
                    print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")

                value = model.getVarByName(f'w_plus[{t},{n},{m},{k}]').X
                if value > 0:
                    print(f"W_plus, Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")

                value = model.getVarByName(f'w_minus[{t},{n},{m},{k}]').X
                if value > 0:
                    print(f"W_minus, Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")

print("\nProduction variables")
for t in problem.traders:
    for n in problem.nodes:
        for m in problem.third_stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'q_production[{t},{n},{m},{k}]').X
                if value > 0:
                    print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")

print("\nSales variables")
for t in problem.traders:
    for n in problem.nodes:
        for m in problem.third_stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'q_sales[{t},{n},{m},{k}]').X
                if value > 0:
                    print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")

print("\nEntry capacity variables")
for t in problem.traders:
    for n in problem.nodes:
        for m in problem.stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'x_plus[{n},{m},{t},{k}]').X
                if value > 0:
                    print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")

print("\nExit capacity variables")
for t in problem.traders:
    for n in problem.nodes:
        for m in problem.stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'x_minus[{n},{m},{t},{k}]').X
                if value > 0:
                    print(f"Trader {t}, Node {n}, Stage {m}, Commodity {k}: {value}")

print("\nSale of entry capacity variables")
for t in problem.traders:
    for n in problem.nodes:
        for m in problem.stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'y_plus[{n},{m},{t},{k}]').X
                if value > 0:
                    print(f"Trader {t}, Node {n}, Stage {m}, Commodity {k}: {value}")

print("\nSale of exit capacity variables")
for t in problem.traders:
    for n in problem.nodes:
        for m in problem.stage_ids:
            for k in problem.commodities:
                value = model.getVarByName(f'y_minus[{n},{m},{t},{k}]').X
                if value > 0:
                    print(f"Trader {t}, Node {n}, Stage {m}, Commodity {k}: {value}")