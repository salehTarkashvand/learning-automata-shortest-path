"""
Learning Automaton Agent for Shortest Path Problems.

This module implements a single Learning Automaton (LA) that can be
assigned to a node in the graph. Each LA maintains a probability
vector over its available actions (edges) and updates these
probabilities using the Linear Reward-Penalty (LRP) scheme.

The LA is intentionally decoupled from the environment: it only knows
about its own action set and the rewards it receives.

Author: Saleh
Date: 2026
"""

import random


class LearningAutomaton:
    """
    A single Learning Automaton using the Linear Reward-Penalty scheme.

    Each automaton is associated with one node in the graph and learns
    which outgoing edge is most likely to lead to the goal with minimal
    cost.

    Attributes:
        node_id (int):                  The graph node this LA controls.
        actions (list):                 Available actions (neighbor node IDs).
        probabilities (dict):           Probability of choosing each action.
        alpha (float):                  Reward learning rate.
        beta (float):                   Penalty learning rate.
        min_probability (float):        Lower bound for action probabilities
                                        to avoid premature convergence.
    """

    def __init__(self, node_id, actions, alpha=0.05, beta=0.05, min_probability=0.01):
        """
        Initialize the Learning Automaton.

        Args:
            node_id (int):          The graph node this LA controls.
            actions (list):         List of neighbor node IDs (valid actions).
            alpha (float):          Reward learning rate (default: 0.05).
            beta (float):           Penalty learning rate (default: 0.05).
            min_probability (float): Minimum probability per action to keep
                                     exploration alive (default: 0.01).
        """
        if not actions:
            raise ValueError(f"No actions provided for node {node_id}.")

        self.node_id = node_id
        self.actions = list(actions)
        self.alpha = alpha
        self.beta = beta
        self.min_probability = min_probability

        # Initialize probabilities uniformly
        n = len(self.actions)
        self.probabilities = {action: 1.0 / n for action in self.actions}

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------
    def choose_action(self):
        """
        Select an action based on the current probability vector.

        Uses roulette-wheel (fitness-proportionate) selection.

        Returns:
            int: The chosen action (neighbor node ID).
        """
        r = random.random()
        cumulative = 0.0
        for action in self.actions:
            cumulative += self.probabilities[action]
            if r <= cumulative:
                return action

        # Fallback for floating-point rounding errors
        return self.actions[-1]

    # ------------------------------------------------------------------
    # Learning updates
    # ------------------------------------------------------------------
    def reward(self, chosen_action):
        """
        Update probabilities after receiving a reward.

        Linear Reward (LR) update:
            p(a) <- p(a) + alpha * (1 - p(a))       for the chosen action
            p(b) <- p(b) - alpha * p(b)             for other actions

        Args:
            chosen_action (int): The action that led to a reward.
        """
        if chosen_action not in self.probabilities:
            return

        for action in self.actions:
            if action == chosen_action:
                self.probabilities[action] += self.alpha * (
                    1.0 - self.probabilities[action]
                )
            else:
                self.probabilities[action] -= self.alpha * self.probabilities[action]

        self._normalize_and_clip()

    def penalize(self, chosen_action):
        """
        Update probabilities after receiving a penalty.

        Linear Penalty (LP) update:
            p(a) <- p(a) - beta * p(a)                       for chosen action
            p(b) <- p(b) + beta * (1/(n-1) - p(b))           for others

        Args:
            chosen_action (int): The action that led to a penalty.
        """
        if chosen_action not in self.probabilities:
            return

        n = len(self.actions)
        if n == 1:
            return  # Nothing to update

        for action in self.actions:
            if action == chosen_action:
                self.probabilities[action] -= self.beta * self.probabilities[action]
            else:
                target = 1.0 / (n - 1)
                self.probabilities[action] += self.beta * (
                    target - self.probabilities[action]
                )

        self._normalize_and_clip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _normalize_and_clip(self):
        """
        Ensure probabilities sum to 1 and respect the minimum bound.

        The clipping prevents any action from being completely excluded,
        which preserves exploration.
        """
        n = len(self.actions)

        # Clip to minimum
        for action in self.actions:
            if self.probabilities[action] < self.min_probability:
                self.probabilities[action] = self.min_probability

        # Re-normalize
        total = sum(self.probabilities.values())
        for action in self.actions:
            self.probabilities[action] /= total

    def get_best_action(self):
        """
        Return the action with the highest probability.

        Returns:
            int: The most likely action.
        """
        return max(self.probabilities, key=self.probabilities.get)

    def get_probabilities(self):
        """
        Return a copy of the current probability vector.

        Returns:
            dict: {action: probability}
        """
        return dict(self.probabilities)

    def reset(self):
        """
        Reset the automaton to a uniform probability distribution.
        """
        n = len(self.actions)
        self.probabilities = {action: 1.0 / n for action in self.actions}

    def __repr__(self):
        """Human-readable representation for debugging."""
        probs = ", ".join(
            f"{a}:{p:.3f}" for a, p in self.probabilities.items()
        )
        return f"LearningAutomaton(node={self.node_id}, [{probs}])"