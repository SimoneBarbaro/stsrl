import math
import random
import numpy as np

import stsrl.slaythespire as sts
from stsrl.game_encoding import StsEncodings


class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class Node:
    def __init__(self, is_chance=False, is_terminal=False):
        self.edges = []
        self.simulation_count = 0
        self.evaluation_sum = 0.0
        self.is_chance = is_chance
        self.is_terminal = is_terminal

    def __str__(self):
        edge_str = '\n-- '.join([str(e).replace("-- ", "---- ") for e in self.edges])
        node_str = "{" + f"visits: {self.simulation_count}, avg_evaluation: {self.evaluation_sum / self.simulation_count if self.simulation_count > 0 else 0}" + "}"
        if len(self.edges) > 0:
            node_str += "children: \n-- " + edge_str
        return node_str

    def is_expanded(self):
        return len(self.edges) > 0

    def avg_value(self, minimax: MinMaxStats) -> float:
        return minimax.normalize(self.evaluation_sum / self.simulation_count if self.simulation_count > 0 else 0)

    def add_edge(self, edge):
        self.edges.append(edge)


class Edge:
    def __init__(self, action, node: Node, description=None):
        self.action = action
        self.node: Node = node
        self.description = description

    def __str__(self):
        return f"{self.description}->{str(self.node)}" if self.description else "->" + str(self.node)


class NodeEvaluator:
    def evaluate_node_state(self, node_state) -> float:
        pass


class ValueModuleEvaluator(NodeEvaluator):
    def __init__(self, value_module):
        self.value_module = value_module

    def evaluate_node_state(self, node_state) -> float:
        if isinstance(node_state, sts.GameContext):
            torch_state = StsEncodings.encode_game(node_state) / StsEncodings.nniInstance.getObservationMaximums()
        else:
            torch_state = StsEncodings.encode_battle(self.gc,self.bc) / StsEncodings.nniInstance.getBattleObservationMaximums()


class BattleNodeEval(NodeEvaluator):
    def evaluate_node_state(self, node_state) -> float:
        reward = 0
        if node_state.outcome == sts.BattleOutcome.PLAYER_VICTORY:
            reward = (1 + (node_state.player.hp / node_state.player.max_hp) + len(node_state.potions) / 5) / 3
        elif node_state.outcome == sts.BattleOutcome.PLAYER_LOSS:
            reward = -1
        return reward


class RolloutPolicy:
    def rollout(self, state) -> any:
        pass


class RandomRolloutPolicy(RolloutPolicy):
    def rollout(self, state) -> any:
        while not state.is_terminal():
            state.execute(random.choice(state.get_available_actions()))
        return state


class NoRolloutPolicy(RolloutPolicy):
    def rollout(self, state) -> any:
        return state


class ExplorationPolicy:
    def exploration_bonus(self, node: Node, edge: Edge) -> float:
        pass


class UpperConfidenceBoundPolicy(ExplorationPolicy):
    def __init__(self, exploration_parameter=math.sqrt(2.0)):
        self.exploration_parameter = exploration_parameter

    def exploration_bonus(self, node: Node, edge: Edge):
        return self.exploration_parameter * np.sqrt(
            np.log(node.simulation_count + 1) / (1 + edge.node.simulation_count))


class MCTS:
    def __init__(self,
                 nodeEvaluator: NodeEvaluator = BattleNodeEval(),
                 rolloutPolicy: RolloutPolicy = RandomRolloutPolicy(),
                 explorationPolicy: ExplorationPolicy = UpperConfidenceBoundPolicy(),
                 num_rollouts=1,
                 chance_sampling_breath=4):
        self.root = Node(is_chance=False, is_terminal=False)
        self.state: sts.BattleContext = None
        self.nodeEvaluator = nodeEvaluator
        self.rolloutPolicy = rolloutPolicy
        self.explorationPolicy = explorationPolicy
        self.reset()
        self.rand_gen = random
        self.num_rollouts = num_rollouts
        self.chance_sampling_breath = chance_sampling_breath

    def reset(self):
        self.min_max_stats = MinMaxStats()
        self.search_path = []
        self.action_stack = []
        self.best_action_sequence = []

    def set_root(self, root_state: sts.BattleContext):
        self.root = Node(False, root_state.is_terminal())
        self.state = root_state
        self._expand(self.root, root_state)

    def search(self, num_simulations):
        self._simulate_multiple(num_simulations)

    def _simulate(self):
        node = self.root
        state = self.state.copy()
        self.search_path = []
        while node.is_expanded() and not state.is_terminal():
            edge_idx = self._select(node, state)
            edge = node.edges[edge_idx]
            node = edge.node

            self.search_path.append(node)
        self._expand(node, state)
        for _ in range(self.num_rollouts):
            reward = self._rollout(state)
            self._backpropagate(node, reward)

    def _simulate_multiple(self, num_simulations):
        for _ in range(num_simulations):
            self._simulate()

    def _select(self, node: Node, state):
        if node.is_terminal:
            return
        if not node.is_expanded():
            self._expand(node, state)

        if len(node.edges) == 1:
            return 0
        if node.simulation_count == 0:
            return random.randrange(0, len(node.edges))

        best_edge = 0
        best_edge_value = self._evaluate_edge(node, best_edge)

        for i in range(0, len(node.edges)):
            value = self._evaluate_edge(node, i)
            if value > best_edge_value:
                best_edge = i
                best_edge_value = value

        return best_edge

    def _evaluate_edge(self, parent: Node, edge_idx):
        edge = parent.edges[edge_idx]
        quality_value = parent.avg_value(self.min_max_stats)
        if parent.is_chance:
            quality_value = 0
        exploration_value = self.explorationPolicy.exploration_bonus(parent, edge)
        return quality_value + exploration_value

    def _expand(self, node, cur_state):
        if not cur_state.is_terminal():
            if node.is_chance:
                self._expand_chance_node(node)
                # logger.debug("Expanded chance node with %d edges", self.chance_sampling_breath)
            else:
                for action in cur_state.get_available_actions():
                    next_state = sts.BattleContext(cur_state)
                    action.execute(next_state)
                    is_chance = not cur_state.is_same_rng_counters(next_state)
                    node.edges.append(Edge(action, Node(is_chance=is_chance, is_terminal=next_state.is_terminal()),
                                           description=action.print_desc(cur_state)))
                # logger.debug("Expanded node with %d edges", len(node.edges))

    def _expand_chance_node(self, node):
        for _ in range(self.chance_sampling_breath):
            node.edges.append(Edge(
                self.rand_gen.randint(1, 100), Node(is_chance=False, is_terminal=False),
                description="SampleRngCounters"))

    def _rollout(self, state):
        final_state = self.rolloutPolicy.rollout(state)
        return self.nodeEvaluator.evaluate_node_state(final_state)

    def _backpropagate(self, node: Node, reward: float):
        for node in self.search_path[::-1]:
            node.simulation_count += 1
            node.evaluation_sum += reward

    def best_action(self):
        # Get action with the highest value
        best_action = None
        best_value = -float('inf')
        for child in self.root.edges:
            if child.node.avg_value(self.min_max_stats) > best_value:
                best_value = child.node.avg_value(self.min_max_stats)
                best_action = child.action
        return best_action
