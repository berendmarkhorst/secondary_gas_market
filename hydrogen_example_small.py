from objects import *
import networkx as nx
import pandas as pd
import copy

# Sets
nodes = [1, 2, 3]
arcs = [(1, 2), (2, 3), (2, 1), (3, 2)]
digraph = nx.DiGraph()
digraph.add_nodes_from(nodes)
digraph.add_edges_from(arcs)

# Objects
t1 = Trader(1, "Trader 1")
traders = [t1]
gas = Commodity(1, "gas")
hydrogen = Commodity(2, "hydrogen")
commodities = [gas, hydrogen]
stage_arcs = [StageArc(id, source, sink, 100, {k: 2 for k in commodities}) for (id, (source, sink)) in enumerate(arcs)]

# Parameters
loss_rate = 0.2
allowed_percentage = 1
gamma = 0.25

# Useful stuff
node_demands3yes = {(1, "gas"): 0, (2, "gas"): 0, (3, "gas"): 3.2,
                 (1, "hydrogen"): 0, (2, "hydrogen"): 1, (3, "hydrogen"): 0}
node_demands3no = {(1, "gas"): 0, (2, "gas"): 4, (3, "gas"): 0,
                 (1, "hydrogen"): 0, (2, "hydrogen"): 2, (3, "hydrogen"): 0}

# Stage 1 and 2
stage_nodes1 = [StageNode(node, f"Node {node}", None, None, None, None, None, {k: 30 for k in commodities}, {k: 30 for k in commodities},
                    {(t, k): 2 for k in commodities for t in traders}, {(t, k): 2 for k in commodities for t in traders}, allowed_percentage) for node in nodes]

# Stage 3
stage_nodes2 = [StageNode(1, "Node 1", {gas: 0, hydrogen: 0}, {(t1, gas):10, (t1, hydrogen): 10}, {(t1, gas):10, (t1, hydrogen): 10}, {(t1, gas):2, (t1, hydrogen): 2}, {(t1, gas):100, (t1, hydrogen): 100}, {k: 30 for k in commodities}, {k: 30 for k in commodities},
                    {(t, k): 2 for k in commodities for t in traders}, {(t, k): 2 for k in commodities for t in traders}, allowed_percentage),
          StageNode(2, "Node 2", {gas: 0, hydrogen: 0}, {(t1, gas): 10, (t1, hydrogen): 10}, {(t1, gas): 0, (t1, hydrogen): 0},
                    {(t1, gas): 2, (t1, hydrogen): 2}, {(t1, gas): 100, (t1, hydrogen): 100}, {k: 30 for k in commodities},
                    {k: 30 for k in commodities},
                    {(t, k): 2 for k in commodities for t in traders}, {(t, k): 2 for k in commodities for t in traders}, allowed_percentage),
          StageNode(3, "Node 3", {gas: 0, hydrogen: 0}, {(t1, gas): 10, (t1, hydrogen): 10}, {(t1, gas): 0, (t1, hydrogen): 0},
                    {(t1, gas): 2, (t1, hydrogen): 2}, {(t1, gas): 100, (t1, hydrogen): 100}, {k: 30 for k in commodities},
                    {k: 30 for k in commodities},
                    {(t, k): 2 for k in commodities for t in traders}, {(t, k): 2 for k in commodities for t in traders}, allowed_percentage),
          ]

stage_nodes3 = [StageNode(1, "Node 1", {gas: 0, hydrogen: 0}, {(t1, gas):10, (t1, hydrogen): 10}, {(t1, gas):10, (t1, hydrogen): 10}, {(t1, gas):2, (t1, hydrogen): 2}, {(t1, gas):100, (t1, hydrogen): 100}, {k: 30 for k in commodities}, {k: 30 for k in commodities},
                    {(t, k): 2 for k in commodities for t in traders}, {(t, k): 2 for k in commodities for t in traders}, allowed_percentage),
          StageNode(2, "Node 2", {gas: 0, hydrogen: 1}, {(t1, gas): 10, (t1, hydrogen): 10}, {(t1, gas): 0, (t1, hydrogen): 0},
                    {(t1, gas): 2, (t1, hydrogen): 2}, {(t1, gas): 100, (t1, hydrogen): 100}, {k: 30 for k in commodities},
                    {k: 30 for k in commodities},
                    {(t, k): 2 for k in commodities for t in traders}, {(t, k): 2 for k in commodities for t in traders}, allowed_percentage),
          StageNode(3, "Node 3", {gas: 3.2, hydrogen: 0}, {(t1, gas): 10, (t1, hydrogen): 10}, {(t1, gas): 0, (t1, hydrogen): 0},
                    {(t1, gas): 2, (t1, hydrogen): 2}, {(t1, gas): 100, (t1, hydrogen): 100}, {k: 30 for k in commodities},
                    {k: 30 for k in commodities},
                    {(t, k): 2 for k in commodities for t in traders}, {(t, k): 2 for k in commodities for t in traders}, allowed_percentage),
          ]

# stage_nodes2 = stage_nodes3
# for n in nodes:
#     stage_nodes2[n-1].node_demands = {gas: 0, hydrogen: 0}

stage1 = Stage(1, "long term", 1, stage_nodes1, stage_arcs, None)
stage2 = Stage(2, "day ahead", 1, stage_nodes2, stage_arcs, stage1)
stage3 = Stage(3, "intra day", 1, stage_nodes3, stage_arcs, stage2)
#
# probability4 = 1
#
# # Arc costs and capacities
# arc_costs = {(arc, k): 10 for arc in arcs for k in commodities}
# arc_capacities = {arc: 30 for arc in arcs}
#
# # Entry and exit costs
# entry_costs3 = {(node, k): 2 for node in nodes for k in commodities}
# exit_costs3 = {(node, k): 2 for node in nodes for k in commodities}
#
# # Node capacities and demands
# node_capacities = {(node, k): 30 for node in nodes for k in commodities}
#
#
# # Production costs and capacities
# production_costs3 = {(1, 1, "gas"): 10, (2, 1, "gas"): 50, (3, 1, "gas"): 20,
#                      (1, 1, "hydrogen"): 10, (2, 1, "hydrogen"): 50, (3, 1, "hydrogen"): 20,
#                      }
# production_capacities3 = {(1, 1, "gas"): 10, (1, 2, "gas"): 0, (1, 3, "gas"): 0,
#                           (1, 1, "hydrogen"): 30, (1, 2, "hydrogen"): 0, (1, 3, "hydrogen"): 30,
#                           }
#
# # Storage costs and capacities
# storage_costs3 = {(1, 1, "gas"): 2, (1, 2, "gas"): 2, (1, 3, "gas"): 2,
#                   (1, 1, "hydrogen"): 2, (1, 2, "hydrogen"): 2, (1, 3, "hydrogen"): 2,
#                   }
# storage_capacities = {(1, 1, "gas"): 100, (1, 2, "gas"): 100, (1, 3, "gas"): 100,
#                       (1, 1, "hydrogen"): 100, (1, 2, "hydrogen"): 100, (1, 3, "hydrogen"): 100,
#                       }

# Third stage: dummy for now
# stage4 = Stage(1, "intra day", arc_costs, entry_costs3, exit_costs3, arc_capacities, node_capacities, probability4, node_demands3no, production_costs3, production_capacities3, storage_costs3, storage_capacities, None)
# stage5 = Stage(2, "intra day", arc_costs, entry_costs3, exit_costs3, arc_capacities, node_capacities, probability4, node_demands3no, production_costs3, production_capacities3, storage_costs3, storage_capacities, stage4)

# Stages objects
stages = [stage1, stage2, stage3]

# Problem object
problem = Problem(digraph, stages, traders, loss_rate, commodities, gamma)

model = problem.build_model()
model.optimize()
#
# print("Flow variables")
for t in problem.traders:
    for a in problem.arcs:
        for m in problem.third_stages:
            for k in problem.commodities:
                value = model.getVarByName(f'f[{t.trader_id},{a[0]},{a[1]},{m.stage_id},{k.commodity_id}]').X
                if value > 0.1:
                    print(f"Trader {t.trader_id}, Arc {a}, Stage {m.stage_id}, Commodity {k.name}: {value}")
#
# print("\nStorage variables")
# for t in problem.traders:
#     for n in problem.nodes:
#         for m in problem.third_stage_ids:
#             for k in problem.commodities:
#                 value = model.getVarByName(f'v[{t},{n},{m},{k}]').X
#                 if value > 0:
#                     print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")
#
#                 value = model.getVarByName(f'w_plus[{t},{n},{m},{k}]').X
#                 if value > 0:
#                     print(f"W_plus, Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")
#
#                 value = model.getVarByName(f'w_minus[{t},{n},{m},{k}]').X
#                 if value > 0:
#                     print(f"W_minus, Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")
#
# print("\nProduction variables")
# for t in problem.traders:
#     for n in problem.nodes:
#         for m in problem.third_stage_ids:
#             for k in problem.commodities:
#                 value = model.getVarByName(f'q_production[{t},{n},{m},{k}]').X
#                 if value > 0:
#                     print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")
#
# print("\nSales variables")
# for t in problem.traders:
#     for n in problem.nodes:
#         for m in problem.third_stage_ids:
#             for k in problem.commodities:
#                 value = model.getVarByName(f'q_sales[{t},{n},{m},{k}]').X
#                 if value > 0:
#                     print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")
#
# print("\nEntry capacity variables")
# for t in problem.traders:
#     for n in problem.nodes:
#         for m in problem.stage_ids:
#             for k in problem.commodities:
#                 value = model.getVarByName(f'x_plus[{n},{m},{t},{k}]').X
#                 if value > 0:
#                     print(f"Trader {t}, Node {n}, Commodity {k}, Stage {m}: {value}")
#
# print("\nExit capacity variables")
# for t in problem.traders:
#     for n in problem.nodes:
#         for m in problem.stage_ids:
#             for k in problem.commodities:
#                 value = model.getVarByName(f'x_minus[{n},{m},{t},{k}]').X
#                 if value > 0:
#                     print(f"Trader {t}, Node {n}, Stage {m}, Commodity {k}: {value}")
#
# print("\nSale of entry capacity variables")
# for t in problem.traders:
#     for n in problem.nodes:
#         for m in problem.stage_ids:
#             for k in problem.commodities:
#                 value = model.getVarByName(f'y_plus[{n},{m},{t},{k}]').X
#                 if value > 0:
#                     print(f"Trader {t}, Node {n}, Stage {m}, Commodity {k}: {value}")
#
# print("\nSale of exit capacity variables")
# for t in problem.traders:
#     for n in problem.nodes:
#         for m in problem.stage_ids:
#             for k in problem.commodities:
#                 value = model.getVarByName(f'y_minus[{n},{m},{t},{k}]').X
#                 if value > 0:
#                     print(f"Trader {t}, Node {n}, Stage {m}, Commodity {k}: {value}")