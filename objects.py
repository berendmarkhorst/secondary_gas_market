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
                 production_capacity: Dict[Tuple[Trader, Commodity], float],
                 storage_costs: Dict[Tuple[Trader, Commodity], float],
                 storage_capacity: Dict[Tuple[Trader, Commodity], float],
                 entry_capacity: Dict[Commodity, float], exit_capacity: Dict[Commodity, float],
                 entry_costs: Dict[Tuple[Trader, Commodity], float], exit_costs: Dict[Tuple[Trader, Commodity], float],
                 allowed_percentage: float):
        self.node_id = node_id
        self.name = name
        self.node_demands = node_demands
        self.production_costs = production_costs
        self.production_capacity = production_capacity
        self.storage_costs = storage_costs
        self.storage_capacity = storage_capacity
        self.entry_capacity = entry_capacity
        self.exit_capacity = exit_capacity
        self.entry_costs = entry_costs
        self.exit_costs = exit_costs
        self.allowed_percentage = allowed_percentage

    def __repr__(self):
        return f"Node {self.node_id}"


class StageArc:
    def __init__(self, arc_id: int, source: int, sink: int, arc_capacity: float, arc_costs: Dict[Commodity, float]):
        self.arc_id = arc_id
        self.source = source
        self.sink = sink
        self.arc_capacity = arc_capacity
        self.arc_costs = arc_costs

    def __repr__(self):
        return f"Arc {self.arc_id}"


class Stage:
    def __init__(self, stage_id: int, name: str, probability: float, nodes: List[StageNode], arcs: List[StageArc],
                 parent: 'Stage'):
        self.stage_id = stage_id
        self.name = name
        self.probability = probability
        self.nodes = nodes
        self.arcs = arcs
        self.parent = parent
        self.all_parents = self.get_all_parents()
        self.ids_all_parents = self.get_ids_all_parents()

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
                 commodities: List[Commodity], gamma: float):
        self.digraph = digraph
        self.nodes = list(digraph.nodes)
        self.arcs = list(digraph.edges)
        self.incoming_arcs = {node: [arc for arc in self.arcs if arc[1] == node] for node in self.nodes}
        self.outgoing_arcs = {node: [arc for arc in self.arcs if arc[0] == node] for node in self.nodes}
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

    def build_model(self):
        model = gp.Model("Stochastic Secondary Energy Market")

        # Decision variables
        x_plus = model.addVars(self.nodes, self.stage_ids, self.trader_ids, self.commodity_ids, name="x_plus", lb=0.0)
        x_minus = model.addVars(self.nodes, self.stage_ids, self.trader_ids, self.commodity_ids, name="x_minus", lb=0.0)
        y_plus = model.addVars(self.nodes, self.stage_ids, self.trader_ids, self.commodity_ids, name="y_plus", lb=0.0)
        y_minus = model.addVars(self.nodes, self.stage_ids, self.trader_ids, self.commodity_ids, name="y_minus", lb=0.0)
        s_plus = model.addVars(self.nodes, self.stage_ids, self.commodity_ids, name="s_plus", lb=0.0)
        s_minus = model.addVars(self.nodes, self.stage_ids, self.commodity_ids, name="s_minus", lb=0.0)
        f = model.addVars(self.trader_ids, self.arcs, self.third_stage_ids, self.commodity_ids, name="f", lb=0.0)
        q_sales = model.addVars(self.trader_ids, self.nodes, self.second_stage_ids + self.third_stage_ids, self.commodity_ids, name="q_sales", lb=0.0)
        q_production = model.addVars(self.trader_ids, self.nodes, self.third_stage_ids, self.commodity_ids, name="q_production", lb=0.0)
        v = model.addVars(self.trader_ids, self.nodes, self.third_stage_ids, self.commodity_ids, name="v", lb=0.0)
        w_plus = model.addVars(self.trader_ids, self.nodes, self.third_stage_ids, self.commodity_ids, name="w_plus", lb=0.0)
        w_minus = model.addVars(self.trader_ids, self.nodes, self.third_stage_ids, self.commodity_ids, name="w_minus", lb=0.0)

        # Objective function
        objective = 0

        # First part of the objective
        for m in self.stages:
            for t in self.traders:
                for k in self.commodities:
                    for n in m.nodes:
                        entry_costs = x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] * n.entry_costs[(t, k)]
                        exit_costs = x_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] * n.exit_costs[(t, k)]
                        objective += self.stages[m.stage_id-1].probability * (entry_costs + exit_costs)

                        if m.name == "intra day":
                            production_costs = q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] * n.production_costs[(t,k)]
                            storage_costs = v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] * n.storage_costs[(t, k)]
                            flow_costs = gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] * m.get_arc(a).arc_costs[k] for a in self.incoming_arcs[n.node_id])

                            objective += self.stages[m.stage_id - 1].probability * (production_costs + storage_costs + flow_costs)

        model.setObjective(objective, gp.GRB.MINIMIZE)

        # TSO constraints
        # Equation 1b
        for m in self.third_stages:
            for a in self.arcs:
                model.addConstr(gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] for t in self.traders for k in self.commodities) <= self.stages[m.stage_id-1].get_arc(a).arc_capacity,
                                name=f"eq1b[{m.stage_id},{a}]")

        # Equation 1c
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    model.addConstr(s_minus[n.node_id, m.stage_id, k.commodity_id] <= n.exit_capacity[k] - gp.quicksum(x_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for t in self.traders for m_tilde in m.all_parents),
                                    name=f"eq1c[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1d
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    model.addConstr(s_plus[n.node_id, m.stage_id, k.commodity_id] <= n.entry_capacity[k] - gp.quicksum(x_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for t in self.traders for m_tilde in m.all_parents),
                                    name=f"eq1d[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1e
        for n in self.stages[0].nodes:
            for k in self.commodities:
                lhs = s_minus[n.node_id, 1, k.commodity_id]
                rhs = n.allowed_percentage * n.exit_capacity[k]
                model.addConstr(lhs <= rhs, name=f"eq1e[{n.node_id},{k.commodity_id}]")

        # Equation 1f
        for n in self.stages[0].nodes:
            for k in self.commodities:
                lhs = s_plus[n.node_id, 1, k.commodity_id]
                rhs = n.allowed_percentage * n.entry_capacity[k]
                model.addConstr(lhs <= rhs, name=f"eq1f[{n.node_id},{k.commodity_id}]")

        # Equation 1g
        for m in self.second_stages + self.third_stages:
            for n in m.nodes:
                for k in self.commodities:
                    model.addConstr(gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id] for t in self.traders) == n.node_demands[k],
                                    name=f"eq1g[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1h
        k1 = [k for k in self.commodities if k.name == "hydrogen"][0]
        k2 = [k for k in self.commodities if k.name == "gas"][0]
        for m in self.third_stages:
            for a in self.arcs:
                lhs = gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k1.commodity_id] for t in self.traders)
                rhs = gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k2.commodity_id] for t in self.traders) * self.gamma
                model.addConstr(lhs <= rhs, name=f"eq1j[{m.stage_id},{a}]")

        # Supplier constraints
        # Equation 1i
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] <= n.production_capacity[(t, k)],
                                        name=f"eq1i[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1j
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        lhs = q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + w_minus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + gp.quicksum((1 - self.loss_rate) * f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] for a in self.incoming_arcs[n.node_id])
                        rhs = q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + w_plus[t.trader_id, n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(f[t.trader_id, a[0], a[1], m.stage_id, k.commodity_id] for a in self.outgoing_arcs[n.node_id])
                        model.addConstr(lhs == rhs, name=f"eq1j[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1k
        k1 = [k for k in self.commodities if k.name == "hydrogen"][0]
        k2 = [k for k in self.commodities if k.name == "gas"][0]
        for m in self.third_stages:
            for n in m.nodes:
                # Hydrogen
                rhs_numerator = gp.quicksum(
                    (1 - self.loss_rate) * f[t.trader_id, a2[0], a2[1], m.stage_id, k1.commodity_id] for a2 in self.incoming_arcs[n.node_id] for t in
                    self.traders)
                rhs_numerator += gp.quicksum(q_production[t.trader_id, n.node_id, m.stage_id, k1.commodity_id] for t in self.traders)
                rhs_numerator += gp.quicksum(w_minus[t.trader_id, n.node_id, m.stage_id, k1.commodity_id] for t in self.traders)

                # Gas
                rhs_denominator = gp.quicksum(
                    (1 - self.loss_rate) * f[t.trader_id, a2[0], a2[1], m.stage_id, k2.commodity_id] for a2 in self.incoming_arcs[n.node_id] for t in
                    self.traders)
                rhs_denominator += gp.quicksum(q_production[t.trader_id, n.node_id, m.stage_id, k2.commodity_id] for t in self.traders)
                rhs_denominator += gp.quicksum(w_minus[t.trader_id, n.node_id, m.stage_id, k2.commodity_id] for t in self.traders)

                for a1 in self.outgoing_arcs[n.node_id]:
                    # Wat gaat er uit?
                    # Hydrogen
                    lhs_numerator = gp.quicksum(f[t.trader_id, a1[0], a1[1], m.stage_id, k1.commodity_id] for t in self.traders)
                    lhs_numerator += gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k1.commodity_id] for t in self.traders)
                    lhs_numerator += gp.quicksum(w_plus[t.trader_id, n.node_id, m.stage_id, k1.commodity_id] for t in self.traders)

                    # Gas
                    lhs_denominator = gp.quicksum(f[t.trader_id, a1[0], a1[1], m.stage_id, k2.commodity_id] for t in self.traders)
                    lhs_denominator += gp.quicksum(q_sales[t.trader_id, n.node_id, m.stage_id, k2.commodity_id] for t in self.traders)
                    lhs_denominator += gp.quicksum(w_plus[t.trader_id, n.node_id, m.stage_id, k2.commodity_id] for t in self.traders)

                    model.addConstr(lhs_numerator * rhs_denominator == lhs_denominator * rhs_numerator, name=f"eq1k[{n.node_id},{m.stage_id},{a1}]")

        # Equation 1l
        for m in self.second_stages + self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_minus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m]) >= q_sales[t.trader_id, n.node_id, m.stage_id, k.commodity_id],
                                        name=f"eq1l[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1m
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(gp.quicksum(x_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] - y_plus[n.node_id, m_tilde.stage_id, t.trader_id, k.commodity_id] for m_tilde in m.all_parents + [m]) >= q_production[t.trader_id, n.node_id, m.stage_id, k.commodity_id],
                                        name=f"eq1m[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1n
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] == gp.quicksum(w_plus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] - w_minus[t.trader_id, n.node_id, m_tilde.stage_id, k.commodity_id] for m_tilde in m.all_parents + [m] if m_tilde.name == "intra day"),
                                        name=f"eq1n[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Equation 1o
        for m in self.third_stages:
            for n in m.nodes:
                for t in self.traders:
                    for k in self.commodities:
                        model.addConstr(v[t.trader_id, n.node_id, m.stage_id, k.commodity_id] <= n.storage_capacity[(t, k)],
                                        name=f"eq1o[{n.node_id},{m.stage_id},{t.trader_id},{k.commodity_id}]")

        # Market constraints
        # Equation 1p
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    lhs = gp.quicksum(x_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    rhs = s_minus[n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(y_minus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1p[{n.node_id},{m.stage_id},{k.commodity_id}]")

        # Equation 1q
        for m in self.stages:
            for n in m.nodes:
                for k in self.commodities:
                    lhs = gp.quicksum(x_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    rhs = s_plus[n.node_id, m.stage_id, k.commodity_id] + gp.quicksum(y_plus[n.node_id, m.stage_id, t.trader_id, k.commodity_id] for t in self.traders)
                    model.addConstr(lhs == rhs, name=f"eq1q[{n.node_id},{m.stage_id},{k.commodity_id}]")

        return model



