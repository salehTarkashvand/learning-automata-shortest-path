"""
Visualization utilities for the Learning Automata Shortest Path project.

This module provides publication-quality visualizations:
    - plot_convergence():      Moving-average success rate over episodes.
    - plot_graph_with_path():  Graph with the learned path highlighted.
    - plot_path_comparison():  Side-by-side comparison of suboptimal vs optimal.
    - plot_probability_bars(): Learned probability vectors per node.

All figures are saved to the `results/` directory.

Author: Saleh
Date: 2026
"""

import os

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


# ----------------------------------------------------------------------
# Layout & style
# ----------------------------------------------------------------------
# Node positions match the original figure (S=1 top-left, D=4 right)
NODE_POSITIONS = {
    1: (0.0, 2.2),
    2: (2.0, 4.0),
    3: (5.0, 3.5),
    4: (6.2, 2.2),
    5: (4.0, 0.0),
    6: (1.5, 0.0),
    7: (0.0, 0.0),
}

START_NODE = 1
END_NODE = 4

COLOR_NODE = "#1f5fdb"          # royal blue (matches original)
COLOR_NODE_START = "#28a745"    # green for start
COLOR_NODE_END = "#ff9800"      # orange for goal
COLOR_EDGE = "#9aa0a6"          # light gray
COLOR_PATH = "#e53935"          # vivid red for optimal path
COLOR_WEIGHT_BG = "#ffffff"
COLOR_TEXT = "#222222"

RESULTS_DIR = "results"


def _ensure_results_dir():
    """Create the results directory if it does not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)


def _build_networkx_graph(graph):
    """Convert the adjacency dict to an undirected networkx graph."""
    G = nx.Graph()
    for u, neighbors in graph.items():
        for v, w in neighbors.items():
            G.add_edge(u, v, weight=w)
    return G


def _path_edges(path):
    """Return a set of normalized (min,max) edges that belong to the path."""
    if not path:
        return set()
    return {(min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            for i in range(len(path) - 1)}


# ======================================================================
# 1. Convergence plot
# ======================================================================
def plot_convergence(success_history, window=100, filename="convergence.png"):
    """Plot the moving-average success rate over training episodes."""
    _ensure_results_dir()

    smoothed = []
    for i in range(len(success_history)):
        start = max(0, i - window + 1)
        chunk = success_history[start:i + 1]
        smoothed.append(sum(chunk) / len(chunk) * 100)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(smoothed, color="#1f77b4", linewidth=2, label=f"{window}-ep moving average")
    ax.fill_between(range(len(smoothed)), smoothed, alpha=0.15, color="#1f77b4")
    ax.axhline(100, color="green", linestyle="--", linewidth=1, alpha=0.6, label="Perfect")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("Learning Progress - Success Rate Over Time",
                 fontsize=14, fontweight="bold")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)

    out_path = os.path.join(RESULTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[Saved] {out_path}")

    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ======================================================================
# 2. Main graph with learned path
# ======================================================================
def plot_graph_with_path(graph, path, title=None, filename="learned_path.png"):
    """
    Draw the graph with the learned path highlighted in red.

    Args:
        graph (dict):   Adjacency list with weights.
        path (list):    Sequence of node IDs representing the path.
        title (str):    Optional custom title.
        filename (str): Output filename inside results/.
    """
    _ensure_results_dir()
    G = _build_networkx_graph(graph)
    path_set = _path_edges(path)

    # --- Edge styling ---
    edge_colors, edge_widths = [], []
    for u, v in G.edges():
        key = (min(u, v), max(u, v))
        if key in path_set:
            edge_colors.append(COLOR_PATH)
            edge_widths.append(5.0)
        else:
            edge_colors.append(COLOR_EDGE)
            edge_widths.append(2.0)

    # --- Node styling ---
    node_colors = []
    for node in G.nodes():
        if node == START_NODE:
            node_colors.append(COLOR_NODE_START)
        elif node == END_NODE:
            node_colors.append(COLOR_NODE_END)
        else:
            node_colors.append(COLOR_NODE)

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw non-path edges first (so path is on top)
    nx.draw_networkx_edges(
        G, NODE_POSITIONS, ax=ax,
        edge_color=edge_colors, width=edge_widths, alpha=0.95,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, NODE_POSITIONS, ax=ax,
        node_color=node_colors, node_size=1400,
        edgecolors="white", linewidths=2.0,
    )

    # Node labels (white bold numbers)
    nx.draw_networkx_labels(
        G, NODE_POSITIONS, ax=ax,
        font_size=14, font_weight="bold", font_color="white",
    )

    # Edge weight labels with a white rounded background
    for (u, v, data) in G.edges(data=True):
        x = (NODE_POSITIONS[u][0] + NODE_POSITIONS[v][0]) / 2
        y = (NODE_POSITIONS[u][1] + NODE_POSITIONS[v][1]) / 2
        ax.text(
            x, y, str(data["weight"]),
            fontsize=12, color=COLOR_TEXT, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc=COLOR_WEIGHT_BG,
                      ec="none", alpha=0.95),
        )

    # S and D labels next to start/goal
    ax.text(
        NODE_POSITIONS[START_NODE][0] - 0.55, NODE_POSITIONS[START_NODE][1] + 0.35,
        "S", fontsize=18, fontweight="bold", color=COLOR_NODE_START,
    )
    ax.text(
        NODE_POSITIONS[END_NODE][0] + 0.35, NODE_POSITIONS[END_NODE][1] + 0.15,
        "D", fontsize=18, fontweight="bold", color=COLOR_NODE_END,
    )

    # --- Title with cost info ---
    if title is None:
        if path:
            cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            title = (f"Learned Shortest Path:  "
                     f"{' -> '.join(map(str, path))}   (Cost = {cost})")
        else:
            title = "Learned Shortest Path (none found)"
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    # --- Legend ---
    legend_items = [
        mpatches.Patch(color=COLOR_NODE_START, label="Start (S)"),
        mpatches.Patch(color=COLOR_NODE_END, label="Goal (D)"),
        mpatches.Patch(color=COLOR_NODE, label="Node"),
        mpatches.Patch(color=COLOR_PATH, label="Shortest Path"),
    ]
    ax.legend(handles=legend_items, loc="lower right", fontsize=11,
              framealpha=0.95)

    ax.axis("off")
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[Saved] {out_path}")

    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ======================================================================
# 3. Side-by-side comparison
# ======================================================================
def plot_path_comparison(graph, learned_path, optimal_path,
                         filename="path_comparison.png"):
    """
    Show a side-by-side comparison of the learned path and the true optimal.

    Args:
        graph (dict):        Adjacency list with weights.
        learned_path (list): Path found by the learning automata.
        optimal_path (list): True shortest path (e.g., via Dijkstra).
        filename (str):      Output filename inside results/.
    """
    _ensure_results_dir()
    G = _build_networkx_graph(graph)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    def _draw(ax, path, color, label):
        path_set = _path_edges(path)
        edge_colors, edge_widths = [], []
        for u, v in G.edges():
            key = (min(u, v), max(u, v))
            if key in path_set:
                edge_colors.append(color)
                edge_widths.append(5.0)
            else:
                edge_colors.append(COLOR_EDGE)
                edge_widths.append(1.8)

        node_colors = []
        for n in G.nodes():
            if n == START_NODE:
                node_colors.append(COLOR_NODE_START)
            elif n == END_NODE:
                node_colors.append(COLOR_NODE_END)
            else:
                node_colors.append(COLOR_NODE)

        nx.draw_networkx_edges(G, NODE_POSITIONS, ax=ax,
                               edge_color=edge_colors, width=edge_widths)
        nx.draw_networkx_nodes(G, NODE_POSITIONS, ax=ax,
                               node_color=node_colors, node_size=1200,
                               edgecolors="white", linewidths=2)
        nx.draw_networkx_labels(G, NODE_POSITIONS, ax=ax,
                                font_size=13, font_weight="bold",
                                font_color="white")
        for (u, v, d) in G.edges(data=True):
            x = (NODE_POSITIONS[u][0] + NODE_POSITIONS[v][0]) / 2
            y = (NODE_POSITIONS[u][1] + NODE_POSITIONS[v][1]) / 2
            ax.text(x, y, str(d["weight"]), fontsize=11, ha="center",
                    va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.9))

        if path:
            cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            subtitle = f"{' -> '.join(map(str, path))}   (Cost = {cost})"
        else:
            subtitle = "No path found"
        ax.set_title(f"{label}\n{subtitle}", fontsize=13, fontweight="bold")
        ax.axis("off")

    _draw(axes[0], learned_path, COLOR_PATH, "Learned Path")
    _draw(axes[1], optimal_path, "#2e7d32", "True Shortest Path (Dijkstra)")

    fig.suptitle("Learned vs. Optimal Path", fontsize=16, fontweight="bold")
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[Saved] {out_path}")

    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ======================================================================
# 4. Probability bars per node
# ======================================================================
def plot_probability_bars(automatons, filename="probabilities.png"):
    """
    Bar chart of learned action probabilities for each node.

    Args:
        automatons (dict): {node_id: LearningAutomaton}
        filename (str):    Output filename inside results/.
    """
    _ensure_results_dir()

    node_ids = sorted(automatons.keys())
    n = len(node_ids)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.5 * rows))
    axes = axes.flatten()

    for i, node_id in enumerate(node_ids):
        ax = axes[i]
        la = automatons[node_id]
        probs = la.get_probabilities()
        actions = sorted(probs.keys())
        values = [probs[a] for a in actions]
        colors = ["#e53935" if v == max(values) else "#1f5fdb" for v in values]

        bars = ax.bar([str(a) for a in actions], values, color=colors,
                      edgecolor="white", linewidth=1.2)
        ax.set_title(f"Node {node_id}", fontweight="bold", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.set_xlabel("Action (neighbor)")
        ax.grid(axis="y", alpha=0.3)

        # Annotate each bar
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=9)

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Learned Action Probabilities per Node",
                 fontsize=15, fontweight="bold")
    out_path = os.path.join(RESULTS_DIR, filename)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"[Saved] {out_path}")

    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)


# ======================================================================
# 5. Dijkstra helper (for comparison)
# ======================================================================
def dijkstra_shortest_path(graph, start, end):
    """
    Compute the true shortest path using Dijkstra's algorithm.

    Args:
        graph (dict): Adjacency list with weights.
        start (int):  Start node.
        end (int):    Goal node.

    Returns:
        tuple: (path, cost)
    """
    G = _build_networkx_graph(graph)
    try:
        path = nx.shortest_path(G, source=start, target=end, weight="weight")
        cost = nx.shortest_path_length(G, source=start, target=end, weight="weight")
        return path, cost
    except nx.NetworkXNoPath:
        return None, float("inf")