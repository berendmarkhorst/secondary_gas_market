from objects import *
import networkx as nx
import pandas as pd

# Sets
nodes = [1, 2, 3]
arcs = [(1, 2), (2, 3), (3, 1), (2, 1), (3, 2), (1, 3)]
digraph = nx.DiGraph()
digraph.add_nodes_from(nodes)
digraph.add_edges_from(arcs)

traders = [1, 2]
commodities = ["gas", "hydrogen"]

# Parameters
loss_rate = 0.2
allowed_percentage = {node: 1 for node in nodes}
probability1 = 1
probability2 = 1
probability3 = 0
probability4 = 1
probability5 = 0.25
probability6 = 0.25
probability7 = 0.25
gamma = 0.25

# Arc costs and capacities
arc_costs = {(arc, k): 10 for arc in arcs for k in commodities}
arc_capacities = {arc: 30 for arc in arcs}

# Entry and exit costs
entry_costs1 = {(node, k): 1 for node in nodes for k in commodities}
entry_costs2 = {(node, k): 2 for node in nodes for k in commodities}
entry_costs3 = {(node, k): 2 for node in nodes for k in commodities}
exit_costs1 = {(node, k): 1 for node in nodes for k in commodities}
exit_costs2 = {(node, k): 2 for node in nodes for k in commodities}
exit_costs3 = {(node, k): 2 for node in nodes for k in commodities}

# Node capacities and demands
node_capacities = {(node, k): 30 for node in nodes for k in commodities}
node_demands1 = {(1, "gas"): 0, (2, "gas"): 0, (3, "gas"): 0,
                 (1, "hydrogen"): 0, (2, "hydrogen"): 0, (3, "hydrogen"): 0}
node_demands2 = {(1, "gas"): 15, (2, "gas"): 0, (3, "gas"): 0,
                 (1, "hydrogen"): 5, (2, "hydrogen"): 0, (3, "hydrogen"): 0}
node_demands3 = {(1, "gas"): 0, (2, "gas"): 15, (3, "gas"): 0,
                 (1, "hydrogen"): 0, (2, "hydrogen"): 5, (3, "hydrogen"): 0}

# Production costs and capacities
trade_rate = 2
production_costs1 = {(1, 1, "gas"): 1, (2, 1, "gas"): 5, (3, 1, "gas"): 2,
                     (1, 1, "hydrogen"): 1, (2, 1, "hydrogen"): 5, (3, 1, "hydrogen"): 2,
                     (1, 2, "gas"): 1*trade_rate, (2, 2, "gas"): 5*trade_rate, (3, 2, "gas"): 2*trade_rate,
                     (1, 2, "hydrogen"): 1*trade_rate, (2, 2, "hydrogen"): 5*trade_rate, (3, 2, "hydrogen"): 2*trade_rate
                     }
production_costs2 = {(1, 1, "gas"): 10, (2, 1, "gas"): 50, (3, 1, "gas"): 20,
                     (1, 1, "hydrogen"): 10, (2, 1, "hydrogen"): 50, (3, 1, "hydrogen"): 20,
                     (1, 2, "gas"): 10*trade_rate, (2, 2, "gas"): 50*trade_rate, (3, 2, "gas"): 20*trade_rate,
                     (1, 2, "hydrogen"): 10*trade_rate, (2, 2, "hydrogen"): 50*trade_rate, (3, 2, "hydrogen"): 20*trade_rate
                     }
production_costs3 = {(1, 1, "gas"): 10, (2, 1, "gas"): 50, (3, 1, "gas"): 20,
                     (1, 1, "hydrogen"): 10, (2, 1, "hydrogen"): 50, (3, 1, "hydrogen"): 20,
                     (1, 2, "gas"): 10*trade_rate, (2, 2, "gas"): 50*trade_rate, (3, 2, "gas"): 20*trade_rate,
                     (1, 2, "hydrogen"): 10*trade_rate, (2, 2, "hydrogen"): 50*trade_rate, (3, 2, "hydrogen"): 20*trade_rate
                     }
production_capacities1 = {(1, 1, "gas"): 0, (1, 2, "gas"): 30, (1, 3, "gas"): 30,
                          (2, 1, "gas"): 0, (2, 2, "gas"): 30, (2, 3, "gas"): 30,
                          (1, 1, "hydrogen"): 0, (1, 2, "hydrogen"): 30, (1, 3, "hydrogen"): 30,
                          (2, 1, "hydrogen"): 0, (2, 2, "hydrogen"): 30, (2, 3, "hydrogen"): 30
                          }
production_capacities2 = {(1, 1, "gas"): 0, (1, 2, "gas"): 30, (1, 3, "gas"): 30,
                          (2, 1, "gas"): 0, (2, 2, "gas"): 30, (2, 3, "gas"): 30,
                          (1, 1, "hydrogen"): 0, (1, 2, "hydrogen"): 30, (1, 3, "hydrogen"): 30,
                          (2, 1, "hydrogen"): 0, (2, 2, "hydrogen"): 30, (2, 3, "hydrogen"): 30
                          } # t, n, k
production_capacities3 = {(1, 1, "gas"): 30, (1, 2, "gas"): 0, (1, 3, "gas"): 30,
                          (2, 1, "gas"): 30, (2, 2, "gas"): 0, (2, 3, "gas"): 30,
                          (1, 1, "hydrogen"): 30, (1, 2, "hydrogen"): 0, (1, 3, "hydrogen"): 30,
                          (2, 1, "hydrogen"): 30, (2, 2, "hydrogen"): 0, (2, 3, "hydrogen"): 30
                          }

# Storage costs and capacities
storage_costs1 = {(1, 1, "gas"): 1, (1, 2, "gas"): 1, (1, 3, "gas"): 1,
                  (2, 1, "gas"): 1, (2, 2, "gas"): 1, (2, 3, "gas"): 1,
                  (1, 1, "hydrogen"): 1, (1, 2, "hydrogen"): 1, (1, 3, "hydrogen"): 1,
                  (2, 1, "hydrogen"): 1, (2, 2, "hydrogen"): 1, (2, 3, "hydrogen"): 1
                  } # {(t, n, k): costs}
storage_costs2 = {(1, 1, "gas"): 2, (1, 2, "gas"): 2, (1, 3, "gas"): 2,
                  (2, 1, "gas"): 2, (2, 2, "gas"): 2, (2, 3, "gas"): 2,
                  (1, 1, "hydrogen"): 2, (1, 2, "hydrogen"): 2, (1, 3, "hydrogen"): 2,
                  (2, 1, "hydrogen"): 2, (2, 2, "hydrogen"): 2, (2, 3, "hydrogen"): 2
                  }
storage_costs3 = {(1, 1, "gas"): 2, (1, 2, "gas"): 2, (1, 3, "gas"): 2,
                  (2, 1, "gas"): 2, (2, 2, "gas"): 2, (2, 3, "gas"): 2,
                  (1, 1, "hydrogen"): 2, (1, 2, "hydrogen"): 2, (1, 3, "hydrogen"): 2,
                  (2, 1, "hydrogen"): 2, (2, 2, "hydrogen"): 2, (2, 3, "hydrogen"): 2
                  }
storage_capacities = {(1, 1, "gas"): 100, (1, 2, "gas"): 100, (1, 3, "gas"): 100,
                      (2, 1, "gas"): 100, (2, 2, "gas"): 100, (2, 3, "gas"): 100,
                      (1, 1, "hydrogen"): 100, (1, 2, "hydrogen"): 100, (1, 3, "hydrogen"): 100,
                      (2, 1, "hydrogen"): 100, (2, 2, "hydrogen"): 100, (2, 3, "hydrogen"): 100
                      } # {(t, n): costs}

# Stages objects
# First stage
stage1 = Stage(1, "long term", arc_costs, entry_costs1, exit_costs1, arc_capacities, node_capacities, probability1, None, production_costs1, production_capacities1, storage_costs1, storage_capacities, None)

# Second stage
stage2 = Stage(2, "day ahead", arc_costs, entry_costs2, exit_costs2, arc_capacities, node_capacities, probability2, None, production_costs2, production_capacities1, storage_costs2, storage_capacities, stage1)
# stage3 = Stage(3, arc_costs, entry_costs3, exit_costs3, arc_capacities, node_capacities, probability3, node_demands3, production_costs3, production_capacities2, storage_costs3, storage_capacities, stage1)

# Third stage: dummy for now
stage4 = Stage(3, "intra day", arc_costs, entry_costs2, exit_costs2, arc_capacities, node_capacities, probability4, node_demands2, production_costs2, production_capacities1, storage_costs2, storage_capacities, stage2)
# stage5 = Stage(5, arc_costs, entry_costs3, exit_costs3, arc_capacities, node_capacities, probability5, node_demands3, production_costs3, production_capacities2, storage_costs3, storage_capacities, stage2)
# stage6 = Stage(6, arc_costs, entry_costs2, exit_costs2, arc_capacities, node_capacities, probability6, node_demands2, production_costs2, production_capacities1, storage_costs2, storage_capacities, stage3)
# stage7 = Stage(7, arc_costs, entry_costs3, exit_costs3, arc_capacities, node_capacities, probability7, node_demands3, production_costs3, production_capacities2, storage_costs3, storage_capacities, stage3)

stages = [stage1, stage2, stage4]
# stages = [stage1, stage2, stage3, stage4, stage5, stage6, stage7]

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
                if value > 0:
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