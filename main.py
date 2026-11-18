"""
Main entry point for the Learning Automata Shortest Path project.

This script ties together:
    - The graph environment (env/graph_env.py)
    - The learning automata agents (agent/learning_automaton.py)
    - The visualizer (utils/visualizer.py)

It runs the learning loop, reports progress, and extracts the
learned shortest path at the end.

Author: Saleh
Date: 2026
"""

import os
import pickle

from env.graph_env import GraphEnvironment
from agent.learning_automaton import LearningAutomaton
from utils.visualizer import plot_convergence, plot_graph_with_path


# ======================================================================
# 1. Graph definition
# ======================================================================
# Format: {node: {neighbor: weight, ...}, ...}
# The graph is undirected, so both directions are listed explicitly.
GRAPH = {
    1: {2: 3, 6: 1, 7: 2},
    2: {1: 3, 3: 2, 5: 5},
    3: {2: 2, 4: 2, 5: 4},
    4: {3: 2, 5: 3},
    5: {2: 5, 3: 4, 4: 3, 6: 4},
    6: {1: 1, 5: 4, 7: 1},
    7: {1: 2, 6: 1},
}

START_NODE = 1
END_NODE = 4

# Training hyperparameters
N_EPISODES = 3000
ALPHA = 0.05
BETA = 0.05
MAX_STEPS = 15
LOG_EVERY = 300


# ======================================================================
# 2. Initialization
# ======================================================================
def build_automatons(graph, alpha, beta):
    """
    Create one Learning Automaton per graph node.

    Args:
        graph (dict): Adjacency list.
        alpha (float): Reward learning rate.
        beta (float):  Penalty learning rate.

    Returns:
        dict: {node_id: LearningAutomaton}
    """
    automatons = {}
    for node_id, neighbors in graph.items():
        automatons[node_id] = LearningAutomaton(
            node_id=node_id,
            actions=list(neighbors.keys()),
            alpha=alpha,
            beta=beta,
        )
    return automatons


# ======================================================================
# 3. Episode runner
# ======================================================================
def run_episode(env, automatons):
    """
    Run a single episode: traverse the graph from start to goal.

    The agent chooses an action at each node using the corresponding
    automaton. The automaton is updated at the END of the episode based
    on the final outcome (success or failure).

    Args:
        env (GraphEnvironment): The environment.
        automatons (dict):      {node_id: LearningAutomaton}

    Returns:
        dict: Episode summary with keys:
            - success (bool)
            - path (list)
            - cost (int)
            - decisions (list of (node, action))
    """
    state = env.reset()
    decisions = []  # Track (node, chosen_action) for credit assignment

    done = False
    while not done:
        automaton = automatons[state]
        action = automaton.choose_action()
        decisions.append((state, action))

        next_state, reward, done, info = env.step(action)
        state = next_state

    success = info.get("success", False)

    # ------------------------------------------------------------------
    # Credit assignment: propagate reward/penalty to all automata
    # that participated in this episode.
    # ------------------------------------------------------------------
        # ------------------------------------------------------------------
    # Credit assignment with quality-based reward
    # ------------------------------------------------------------------
    OPTIMAL_COST = 7    # known optimal cost for this graph
    WORST_COST = 15     # a reasonable upper bound for a successful path

    if success:
        cost = info.get("total_cost", WORST_COST)
        # Quality in [0, 1]: 1.0 = optimal, 0.0 = worst acceptable
        quality = max(
            0.0, min(1.0, (WORST_COST - cost) / (WORST_COST - OPTIMAL_COST))
        )

        for node, action in decisions:
            if quality >= 0.5:
                automatons[node].reward(action)
            else:
                automatons[node].penalize(action)
    else:
        # Failure: penalize the last decision that caused it
        if decisions:
            last_node, last_action = decisions[-1]
            automatons[last_node].penalize(last_action)

    return {
        "success": success,
        "path": info.get("path", env.visited_nodes),
        "cost": info.get("total_cost", env.total_cost),
        "decisions": decisions,
    }


# ======================================================================
# 4. Training loop
# ======================================================================
def train(env, automatons, n_episodes, log_every=300):
    """
    Train the automatons over multiple episodes.

    Args:
        env (GraphEnvironment): The environment.
        automatons (dict):      {node_id: LearningAutomaton}
        n_episodes (int):       Number of episodes to run.
        log_every (int):        Print progress every N episodes.

    Returns:
        dict: Training statistics:
            - success_history (list): 1 if episode succeeded else 0
            - cost_history (list):    Total cost of each successful episode
            - best_path (list):       Shortest path found during training
            - best_cost (float):      Cost of the best path
    """
    success_history = []
    cost_history = []
    best_path = None
    best_cost = float("inf")

    for episode in range(1, n_episodes + 1):
        result = run_episode(env, automatons)
        success_history.append(1 if result["success"] else 0)

        if result["success"]:
            cost_history.append(result["cost"])
            if result["cost"] < best_cost:
                best_cost = result["cost"]
                best_path = list(result["path"])

        if episode % log_every == 0:
            recent = success_history[-log_every:]
            success_rate = sum(recent) / len(recent) * 100
            print(
                f"Episode {episode:5d} | "
                f"Success rate (last {log_every}): {success_rate:5.1f}% | "
                f"Best cost so far: {best_cost}"
            )

    return {
        "success_history": success_history,
        "cost_history": cost_history,
        "best_path": best_path,
        "best_cost": best_cost,
    }


# ======================================================================
# 5. Greedy path extraction
# ======================================================================
def extract_greedy_path(automatons, start_node, end_node, max_steps=15):
    """
    Extract the path implied by the learned probability vectors.

    At each node, pick the action with the highest probability.

    Args:
        automatons (dict): {node_id: LearningAutomaton}
        start_node (int):  Start node.
        end_node (int):    Goal node.
        max_steps (int):   Safety cap.

    Returns:
        list: Node sequence, or None if the greedy path gets stuck.
    """
    path = [start_node]
    current = start_node

    for _ in range(max_steps):
        if current == end_node:
            return path

        automaton = automatons[current]
        next_node = automaton.get_best_action()

        if next_node in path:
            return None  # Greedy path loops -> fail

        path.append(next_node)
        current = next_node

    return None


# ======================================================================
# 6. Main
# ======================================================================
def main():
    print("=" * 60)
    print("Learning Automata - Shortest Path")
    print("=" * 60)
    print(f"Start node: {START_NODE}, End node: {END_NODE}")
    print(f"Episodes  : {N_EPISODES}")
    print(f"Alpha/Beta: {ALPHA}/{BETA}")
    print("=" * 60)

    # Build environment and agents
    env = GraphEnvironment(
        graph=GRAPH,
        start_node=START_NODE,
        end_node=END_NODE,
        max_steps=MAX_STEPS,
    )
    automatons = build_automatons(GRAPH, ALPHA, BETA)

    # Train
    stats = train(env, automatons, N_EPISODES, log_every=LOG_EVERY)

    # --------------------------------------------------------------
    # Report results
    # --------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Training finished")
    print("=" * 60)
    print(f"Best path found during training: {stats['best_path']}")
    print(f"Best cost                      : {stats['best_cost']}")

    greedy_path = extract_greedy_path(automatons, START_NODE, END_NODE)
    print(f"Greedy path after learning     : {greedy_path}")

    # Print learned probabilities
    print("\nLearned probabilities per node:")
    for node_id in sorted(automatons.keys()):
        la = automatons[node_id]
        probs = la.get_probabilities()
        prob_str = ", ".join(
            f"{a}:{p:.3f}" for a, p in sorted(probs.items())
        )
        print(f"  Node {node_id}: {prob_str}")

    # --------------------------------------------------------------
    # Persist results
    # --------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    with open("results/learned_automatons.pkl", "wb") as f:
        pickle.dump(
            {nid: la.get_probabilities() for nid, la in automatons.items()}, f
        )
    print("\n[Saved] results/learned_automatons.pkl")

    # --------------------------------------------------------------
    # Visualization
    # --------------------------------------------------------------
    try:
        # 1. Convergence curve
        plot_convergence(stats["success_history"], window=100)

        # 2. Main graph with learned path
        plot_graph_with_path(GRAPH, greedy_path or stats["best_path"])

        # 3. True optimal path via Dijkstra for comparison
        optimal_path, optimal_cost = dijkstra_shortest_path(
            GRAPH, START_NODE, END_NODE
        )
        print(f"\nDijkstra optimal path: {optimal_path}  (cost = {optimal_cost})")

        # 4. Side-by-side comparison
        plot_path_comparison(
            GRAPH,
            learned_path=greedy_path or stats["best_path"],
            optimal_path=optimal_path,
        )

        # 5. Probability bars per node
        plot_probability_bars(automatons)

    except Exception as e:
        print(f"\n[Warning] Visualization skipped: {e}")


if __name__ == "__main__":
    main()