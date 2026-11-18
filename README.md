# Learning Automata — Shortest Path

Solving the **shortest path problem** on a weighted graph using **Learning Automata (LA)** with the **Linear Reward–Penalty (LRP)** scheme.

Each graph node hosts a learning automaton that learns which outgoing edge is most likely to lead to the goal with minimal cost. The project includes a Gym-like environment, per-node agents, convergence plots, and a side-by-side comparison against Dijkstra's algorithm.

---

## 📌 Problem Statement

Given a weighted, undirected graph with a start node **S** and a goal node **D**, find the shortest path from **S** to **D** without hard-coding any graph-specific knowledge. The agents must *learn* the path purely through trial and error, guided by a reward/penalty signal.

### Graph used in this project


Edge weights are shown in parentheses on the edges. The true shortest path is **1 → 2 → 3 → 4** with total cost **7**.

---

## 🧠 Approach

### Learning Automata (LA)

A Learning Automaton is a simple reinforcement-learning agent that:

1. Maintains a **probability vector** over a set of actions.
2. **Selects** an action according to that distribution.
3. Receives a **reward** or **penalty** from the environment.
4. **Updates** its probabilities using the **Linear Reward–Penalty (LRP)** rule:

**On reward** (action `a` was chosen):

where `α` and `β` are the reward and penalty learning rates.

### Environment

The environment is a Gym-like `GraphEnvironment` that:

- Holds the graph structure.
- Validates agent actions.
- Detects loops, invalid moves, and timeouts.
- Returns a scalar reward and a terminal flag.

### Credit Assignment

At the end of each episode:

- If the path is **near-optimal** (`quality ≥ 0.9`), all decisions along the path are **rewarded**.
- Otherwise, decisions are **penalized**, forcing the agents to explore shorter alternatives.

This quality-based reward shaping is crucial for escaping local optima such as `1 → 6 → 5 → 4` (cost 8) in favor of the true optimum `1 → 2 → 3 → 4` (cost 7).

---

## 📂 Project Structure

---

## ⚙️ Installation

### Option A: Conda (recommended on Apple Silicon)

```bash
conda create -n la-project python=3.11 -c conda-forge -y
conda activate la-project
conda install networkx matplotlib numpy -c conda-forge -y