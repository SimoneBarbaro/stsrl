import math
import time
import random

import numpy as np
import slaythespire as sts

import concurrent.futures


class Node:
    def __init__(self, state:sts.BattleContext, parent=None, is_chance=False):
        self.state = state
        self.parent = parent
        # self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0.0
        self.available_actions = set(state.get_available_actions()) if not is_chance and self.state.outcome == sts.BattleOutcome.UNDECIDED else set()
        self.is_chance = is_chance

    def is_terminal(self):
        return self.state.outcome != sts.BattleOutcome.UNDECIDED

    def expanded(self):
        return len(self.children) > 0

    def avg_value(self) -> float:
        return 0 if self.visits == 0 else self.value / self.visits
    
    def __str__(self):
        actions_sts = []
        if self.is_chance:
            actions_sts = [f"Chance {i}" for i in self.children.keys()]
        else:
            actions_sts = [a.print_desc(self.state) for a in self.children.keys()]
        return f"Node: Visits={self.visits}, Value={self.value:.2f}, AvgValue={self.avg_value():.2f}, Actions={actions_sts}"


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


class MCTS:
    def __init__(self, timeout=1.0,
                 num_simulations=1000,
                 max_workers=4,
                 rollout_policy=random.choice,
                 chance_sampling_breath=4):
        self.root = None
        self.state = None
        self.timeout = timeout
        self.max_workers = max_workers
        self.num_simulations = num_simulations
        self._rollout_policy = rollout_policy
        self.explorationParameter = math.sqrt(2.0)
        self.chance_sampling_breath = chance_sampling_breath
        self.min_max_stats = MinMaxStats()
        self.search_path = []
        self.action_stack = []

    def set_root(self, root_state:sts.BattleContext):
        self.root = Node(root_state)
        self.state = root_state
        self._expand(self.root)

    def search(self):
        self._simulate_multiple(self.num_simulations)

    def _simulate(self):
        node = self.root
        state = sts.BattleContext(node.state)
        while node.expanded() and not node.is_terminal():
            action = self._select(node)
            node = node.children[action]

            self.search_path.append(node)
        self._expand(node)
        reward = self._rollout(state)
        self._backpropagate(node, reward)

    def _simulate_multiple(self, num_simulations):
        for _ in range(num_simulations):
            self._simulate()

    def _select(self, node):
        if not node.expanded():
            self._expand(node)
        best_action = None
        best_score = -float('inf')
        for action in node.children.keys():
            child = node.children[action]
            quality_value = child.avg_value()
            exploration_value = self.explorationParameter * math.sqrt(math.log(node.visits+1) / (child.visits+1))
            if best_score < quality_value + exploration_value:
                best_score = quality_value + exploration_value
                best_action = action

        return best_action

    def _expand(self, node):
        if not node.is_terminal():
            if node.is_chance:
                for i in range(self.chance_sampling_breath):
                    next_state = sts.BattleContext(node.state)
                    next_state.randomize_rng_counters(i)
                    next_state.execute_actions()
                    node.children[i] = Node(next_state, parent=node, is_chance=False)
            else:
                for action in node.state.get_available_actions():
                    next_state = sts.BattleContext(node.state)
                    action.execute(next_state)
                    is_chance = not node.state.is_same_rng_counters(next_state)
                    if is_chance:
                        next_state = sts.BattleContext(node.state)
                        action.submit(next_state)

                    node.children[action] = Node(next_state, parent=node, is_chance=is_chance)

    def _rollout(self, state):
        while state.outcome == sts.BattleOutcome.UNDECIDED:
            action = self._rollout_policy(state.get_available_actions())
            action.execute(state)
        return self._evaluate(state)

    def _backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    def _evaluate(self, state):
        # TODO here will go ML evaluation
        if state.outcome == sts.BattleOutcome.PLAYER_VICTORY:
            return 1.0 + (state.player.hp / state.player.max_hp)
        elif state.outcome == sts.BattleOutcome.PLAYER_LOSS:
            return -1.0
        else:
            return 0.0
            cards_in_hand = len(state.hand)
            energy_penalty = (state.player.energy) * 0.2 if cards_in_hand > 0 else 0
            alive_score = state.monsters_alive_count * 0.5
            return (state.player.hp / state.player.max_hp) - alive_score - energy_penalty

    def best_action(self):
        # Get action with the highest value
        best_action = None
        best_value = -float('inf')
        for action, child in self.root.children.items():
            if child.avg_value() > best_value:
                best_value = child.value
                best_action = action
        return best_action
