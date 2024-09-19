import networkx as nx
import gurobipy as gp
from typing import List, Dict, Tuple


class Commodity:
    def __init__(self, commodity_id: int, name: str):
        self.commodity_id = commodity_id
        self.name = name


class Trader:
    def __init__(self, trader_id: int, name: str):
        self.trader_id = trader_id
        self.name = name


class StageNode:
    def __init__(self, node_id: int, name: str,
                 node_demands: Dict[Commodity, float], production_costs: Dict[Tuple[Trader, Commodity], float],
                 production_capacities: Dict[Tuple[Trader, Commodity], float],
                 tso_entry_costs: Dict[Commodity, float], tso_exit_costs: Dict[Commodity, float],
                 storage_costs: Dict[Tuple[Trader, Commodity], float],
                 storage_capacity: Dict[Tuple[Trader, Commodity], float],
                 entry_costs: Dict[Tuple[Trader, Commodity], float], exit_costs: Dict[Tuple[Trader, Commodity], float],
                 allowed_percentage: float):
        self.node_id = node_id
        self.name = name
        self.node_demands = node_demands
        self.production_costs = production_costs
        self.production_capacity = production_capacities
        self.tso_entry_costs = tso_entry_costs
        self.tso_exit_costs = tso_exit_costs
        self.storage_costs = storage_costs
        self.storage_capacity = storage_capacity
        self.entry_costs = entry_costs
        self.exit_costs = exit_costs
        self.allowed_percentage = allowed_percentage

    def __repr__(self):
        return f"Node {self.node_id}"


class StageArc:
    def __init__(self, arc_id: int, name: str, source: int, sink: int, arc_capacity: float, arc_costs: Dict[Commodity, float]):
        self.arc_id = arc_id
        self.name = name
        self.source = source
        self.sink = sink
        self.arc_capacity = arc_capacity
        self.arc_costs = arc_costs

    def __repr__(self):
        return f"Arc {self.arc_id}"


class Stage:
    def __init__(self, stage_id: int, name: str, probability: float, nodes: List[StageNode], arcs: List[StageArc],
                 parent: 'Stage', hour: int):
        self.stage_id = stage_id
        self.name = name
        self.probability = probability
        self.nodes = nodes
        self.arcs = arcs
        self.parent = parent
        self.all_parents = self.get_all_parents()
        self.ids_all_parents = self.get_ids_all_parents()
        self.hour = hour

    def __repr__(self):
        return f"Stage {self.stage_id}"

    def get_all_parents(self):
        all_parents = []
        parent = self.parent
        while parent is not None:
            all_parents.append(parent)
            parent = parent.parent
        return all_parents

    def get_ids_all_parents(self):
        return [parent.stage_id for parent in self.all_parents]

    def get_arc(self, arc: Tuple[int, int]):
        """Returns the Arc object with the given source and sink."""
        return [a for a in self.arcs if (a.source, a.sink) == arc][0]


class Problem:
    def __init__(self, digraph: nx.Graph, stages: List[Stage], traders: List[Trader], loss_rate: float,
                 commodities: List[Commodity], gamma: float, d_dict: Dict[Commodity, List[str]]):
        self.digraph = digraph
        self.nodes = list(digraph.nodes)
        self.arcs = list(digraph.edges)
        self.incoming_arcs = {digraph.nodes()[node]["ID"]: [arc for arc in self.arcs if arc[1] == node] for node in self.nodes}
        self.outgoing_arcs = {digraph.nodes()[node]["ID"]: [arc for arc in self.arcs if arc[0] == node] for node in self.nodes}
        self.stages = stages
        self.stage_ids = [stage.stage_id for stage in self.stages]  # Starts at 1!
        self.stage_ids_star = self.stage_ids[1:]  # Without 1!
        self.first_stage_id = [stage.stage_id for stage in self.stages if stage.name == "long term"][0]
        self.second_stages = [stage for stage in self.stages if stage.name == "day ahead"]
        self.second_stage_ids = [stage.stage_id for stage in self.second_stages]
        self.third_stages = [stage for stage in self.stages if stage.name == "intra day"]
        self.third_stage_ids = [stage.stage_id for stage in self.third_stages]
        self.traders = traders
        self.trader_ids = [trader.trader_id for trader in self.traders]
        self.loss_rate = loss_rate
        self.commodities = commodities
        self.commodity_ids = [commodity.commodity_id for commodity in self.commodities]
        self.gamma = gamma
        self.node_ids = [digraph.nodes()[node]["ID"] for node in digraph.nodes()]
        self.d_dict = d_dict
        self.d_list = set([item for d in d_dict.values() for item in d])
        self.k_dict = {d: [] for d in self.d_list}
        for k, d_all in self.d_dict.items():
            for d in d_all:
                self.k_dict[d].append(k)

    def build_model(self):
        model = gp.Model("Stochastic Secondary Energy Market")

        # Decision variables
        x_plus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="x_plus", lb=0.0)
        x_minus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="x_minus", lb=0.0)
        y_plus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="y_plus", lb=0.0)
        y_minus = model.addVars(self.node_ids, self.stage_ids, self.trader_ids, self.commodity_ids, name="y_minus", lb=0.0)
        s_plus = model.addVars(self.node_ids, self.stage_ids, self.commodity_ids, name="s_plus", lb=0.0)
        s_minus = model.addVars(self.node_ids, self.stage_ids, self.commodity_ids, name="s_minus", lb=0.0)
        f = model.addVars(self.trader_ids, self.arcs, self.third_stage_ids, self.commodity_ids, name="f", lb=0.0)
        q_sales = model.addVars(self.trader_ids, self.node_ids, self.second_stage_ids + self.third_stage_ids, self.commodity_ids, self.d_list, name="q_sales", lb=0.0)
        q_production = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="q_production", lb=0.0)
        v = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="v", lb=0.0)
        w_plus = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="w_plus", lb=0.0)
        w_minus = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="w_minus", lb=0.0)
        Q = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="Q", lb=0.0)
        W = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="W", lb=0.0)
        surplus_entry = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="surplus_entry", lb=0.0)
        surplus_exit = model.addVars(self.trader_ids, self.node_ids, self.third_stage_ids, self.commodity_ids, name="surplus_exit", lb=0.0)
        entry_capacity = model.addVars(self.node_ids, self.stage_ids, self.commodity_ids, name="entry_capacity", lb=0)
        exit_capacity = model.addVars(self.node_ids, self.stage_ids, self.commodity_ids, name="exit_capacity", lb=0)

        # Objective function
        objective = 0

        # First part of the objective
        for m in self.stages:
            for t in self.traders:
                for k in self.commodities:
                    for n in m.nodes:
                        supplier_entry_costs = x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] * n.entry_costs[(t, k)]
                        supplier_exit_costs = x_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] * n.exit_costs[(t, k)]
                        tso_entry_costs = entry_capacity[n.node_id, m.stage_id, k.commodity_id] * n.tso_entry_costs[k]
                        tso_exit_costs = entry_capacity[n.node_id, m.stage_id, k.commodity_id] * n.tso_exit_costs[k]
                        objective += self.stages[m.stage_id-1].probability * (supplier_entry_costs + supplier_exit_costs + tso_entry_costs + tso_exit_costs)

                        if m.name == "intra day":
                            production_costs = q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] * n.production_costs[(t,k)]
                            storage_costs = v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] * n.storage_costs[(t, k)]
                            flow_costs = gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] * m.get_arc(a).arc_costs[k] for a in self.incoming_arcs[n.node_id])

                            # surplus_entry = q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] - gp.quicksum(x_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m])
                            # surplus_exit = gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for d in self.d_dict[k]) - gp.quicksum(x_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m])

                            objective += self.stages[m.stage_id - 1].probability * (production_costs + storage_costs + flow_costs + (surplus_entry[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + surplus_exit[t.trader_id, n.node_id, m.stage_id, k.commodity_id]) * 100000)

        model.setObjective(objective, gp.GRB.MINIMIZE)

        # NEW CONSTRAINT
        for m in self.third_stages:
            for t in self.traders:
                for k in self.commodities:
                    for n in m.nodes:
                        model.addConstr(surplus_entry[t.trader_id, n.node_id, m.stage_id, k.commodity_id] >= q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] - gp.quicksum(x_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m]))
                        model.addConstr(surplus_exit[t.trader_id, n.node_id, m.stage_id, k.commodity_id] >= gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for d in self.d_dict[k]) - gp.quicksum(x_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m]))

        # TSO constraints
        # Equation 1b
        for m in self.third_stages:
            for a in self.arcs:
                model.addConstr(gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] for t in self.traders for k in self.commodities) <= self.stages[m.stage_id-1].get_arc(a).arc_capacity,
                                name=f"eq1b[{m.stage_id},{a}]")

        # Equation 1c
        for m in self.second_stages + self.third_stages:
            for n in m.nodes:
                for k in self.commodities:
                    model.addConstr(s_minus[n.node_id, m.stage_id, k.commodity_id] <= exit_capacity[n.node_id, m.stage_id, k.commodity_id] - gp.quicksum(x_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for t in self.traders for m_tilde in m.all_parents),
                                    name=f"eq1c[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1d
        for m in self.second_stages + self.third_stages:
            for n in m.nodes:
                for k in self.commodities:
                    model.addConstr(s_plus[n.node_id, m.stage_id, k.commodity_id] <= entry_capacity[n.node_id, m.stage_id, k.commodity_id] - gp.quicksum(x_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for t in self.traders for m_tilde in m.all_parents),
                                    name=f"eq1d[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1e
        for n in self.stages[0].nodes:
            for k in self.commodities:
                lhs = s_minus[n.node_id, 1, k.commodity_id]
                rhs = n.allowed_percentage * exit_capacity[n.node_id, m.stage_id, k.commodity_id]
                model.addConstr(lhs <= rhs, name=f"eq1e[{n.node_id},{k.commodity_id}]")

        # Equation 1f
        for n in self.stages[0].nodes:
            for k in self.commodities:
                lhs = s_plus[n.node_id, 1, k.commodity_id]
                rhs = n.allowed_percentage * entry_capacity[n.node_id, m.stage_id, k.commodity_id]
                model.addConstr(lhs <= rhs, name=f"eq1f[{n.node_id},{k.commodity_id}]")

        # Equation 1g
        for m in self.second_stages + self.third_stages:
            for n in m.nodes:
                for d in self.d_list:
                    model.addConstr(gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for t in self.traders for k in self.k_dict[d]) == n.node_demands[d],
                                    name=f"eq1g[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1h
        k1 = [k for k in self.commodities if k.name == "hydrogen"][0]
        k2 = [k for k in self.commodities if k.name == "gas"][0]
        for m in self.third_stages:
            for a in self.arcs:
                lhs = gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k1.commodity_id] for t in self.traders)
                rhs = gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k2.commodity_id] for t in self.traders) * self.gamma
                model.addConstr(lhs <= rhs, name=f"eq1j[{m.stage_id},{a}]")

        # Equation 1i
        d1 = "pure_hydrogen"
        d2 = "gas_or_mix"
        for m in self.third_stages:
            for n in m.nodes:
                lhs = gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k1.commodity_id, d1] for t in self.traders for k in self.k_dict[d1])
                rhs = gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k2.commodity_id, d2] for t in self.traders for k in self.k_dict[d2]) * self.gamma
                model.addConstr(lhs <= rhs, name=f"1i[{m.stage_id},{n.node_id}]")

        # Supplier constraints
        # Equation 1j
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] <= n.production_capacity[(t, k)],
                                        name=f"eq1j[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1k
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        lhs = q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + w_minus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + gp.quicksum((1 - self.loss_rate) * f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] for a in self.incoming_arcs[n.node_id])
                        rhs = gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for d in self.d_dict[k]) + w_plus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] for a in self.outgoing_arcs[n.node_id])
                        model.addConstr(lhs == rhs, name=f"eq1k[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1l
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(Q[t.trader_id, n.node_id, m.stage_id, k.commodity_id] >= q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] - gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for d in self.d_dict[k]),
                                        name=f"1l[{m.stage_id},{n.node_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1m
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(W[t.trader_id, n.node_id, m.stage_id, k.commodity_id] >= w_minus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] - w_plus[t.trader_id, n.node_id, m.stage_id, k.commodity_id],
                                        name=f"1m[{m.stage_id},{n.node_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1n
        k1 = [k for k in self.commodities if k.name == "hydrogen"][0]
        k2 = [k for k in self.commodities if k.name == "gas"][0]
        for m in self.third_stages:
            for n in m.nodes:
                # Hydrogen
                rhs_numerator = gp.quicksum(f[t.trader_id, a2[0], a2[1], m.stage_id, k1.commodity_id] for a2 in self.incoming_arcs[n.node_id] for t in
                    self.traders)
                rhs_numerator += gp.quicksum(Q[t.trader_id, n.node_id, m.stage_id, k1.commodity_id] for t in self.traders)
                rhs_numerator += gp.quicksum(W[t.trader_id, n.node_id, m.stage_id, k1.commodity_id] for t in self.traders)

                # Gas
                rhs_denominator = gp.quicksum(f[t.trader_id, a2[0], a2[1], m.stage_id, k2.commodity_id] for a2 in self.incoming_arcs[n.node_id] for t in
                    self.traders)
                rhs_denominator += gp.quicksum(Q[t.trader_id, n.node_id, m.stage_id, k2.commodity_id] for t in self.traders)
                rhs_denominator += gp.quicksum(W[t.trader_id, n.node_id, m.stage_id, k2.commodity_id] for t in self.traders)

                for a1 in self.outgoing_arcs[n.node_id]:
                    # Hydrogen
                    lhs_numerator = gp.quicksum(f[t.trader_id, a1[0], a1[1], m.stage_id, k1.commodity_id] for t in self.traders)

                    # Gas
                    lhs_denominator = gp.quicksum(f[t.trader_id, a1[0], a1[1], m.stage_id, k2.commodity_id] for t in self.traders)

                    model.addConstr(lhs_numerator * rhs_denominator == lhs_denominator * rhs_numerator, name=f"eq1n[{n.node_id},{m.stage_id},{a1}]")

        # Equation 1o
        for m in self.second_stages + self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m]) >= gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id, d] for d in self.d_dict[k]),
                                        name=f"eq1o[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1p
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m]) >= q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id],
                                        name=f"eq1p[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1q
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] == gp.quicksum(w_plus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] - w_minus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] for m_tilde in m.all_parents + [m] if m_tilde.name == "intra day"),
                                        name=f"eq1q[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1r
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] <= n.storage_capacity[(t, k)],
                                        name=f"eq1r[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # # # Equation NEW
        # for m in self.stages:
        #     for n in m.nodes:
        #         for t in self.traders:
        #             for k in self.commodities:
        #                 lhs = y_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id]
        #                 rhs = gp.quicksum(x_plus[n.node_id, p.stage_id, t.trader_id, k.commodity_id] - y_plus[n.node_id, p.stage_id, t.trader_id, k.commodity_id] for p in m.all_parents) # + x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id]
        #                 model.addConstr(lhs <= rhs, name=f"eqNEW1[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")
        #
        # # Equation NEW
        # for m in self.stages:
        #     for n in m.nodes:
        #         for t in self.traders:
        #             for k in self.commodities:
        #                 lhs = y_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id]
        #                 rhs = gp.quicksum(x_minus[n.node_id, p.stage_id, t.trader_id, k.commodity_id] - y_minus[n.node_id, p.stage_id, t.trader_id, k.commodity_id] for p in m.all_parents) # + x_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id]
        #                 model.addConstr(lhs <= rhs, name=f"eqNEW2[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Market constraints
        # Equation 1s
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    lhs = gp.quicksum(x_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    rhs = s_minus[n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(y_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1s[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1t
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    lhs = gp.quicksum(x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    rhs = s_plus[n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(y_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1t[{n.node_id},{m.stage_id},{k.commodity_id}]")

        return model



