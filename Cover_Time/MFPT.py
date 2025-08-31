import csv
import random
from collections import defaultdict

import xgi
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------- 全局字体配置 ----------------------
rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,
    "mathtext.fontset": "stix",
    "font.serif": ["SimSun"],
    "axes.unicode_minus": False,
    "font.weight": "bold",
})

# ---------------------- 构建超图 ----------------------
def build_hypergraph(csv_path: str) -> xgi.Hypergraph:
    H = xgi.Hypergraph()
    with open(csv_path, "r", encoding="ISO-8859-1") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            authors_field = (row.get("Authors") or "").strip()
            if not authors_field:
                continue
            authors = [a.strip() for a in authors_field.split(",") if a.strip()]
            H.add_edge(authors, id=f"{row['Title']}_{idx}")
    print(f"Total number of hyperedges: {H.num_edges}")
    print(f"Total number of nodes (authors): {H.num_nodes}\n")
    return H

# ---------------------- 计算权重与超度 ----------------------
def compute_weights(H: xgi.Hypergraph) -> defaultdict:
    weights = defaultdict(lambda: defaultdict(int))
    for _, edge_nodes in H.edges.members(dtype=dict).items():
        deg = len(edge_nodes)
        for i in edge_nodes:
            for j in edge_nodes:
                if i != j:
                    weights[i][j] += deg - 1
    return weights

def compute_hyperdegrees(weights: dict) -> dict:
    return {i: sum(neigh.values()) for i, neigh in weights.items()}

# ---------------------- 重置随机游走 ----------------------
def perform_restart_random_walk(start_node, H, weights, restart_prob=0.05) -> int:
    current = start_node
    visited, steps, total_nodes = set(), 0, len(H.nodes)

    while len(visited) < total_nodes:
        visited.add(current)
        if random.random() < restart_prob:
            current = start_node
        else:
            neighbors = weights[current]
            total_w = sum(neighbors.values())
            probs = [w / total_w for w in neighbors.values()]
            current = random.choices(list(neighbors.keys()), probs)[0]
        steps += 1
    return steps

# ---------------------- 主流程 ----------------------
def main():
    H = build_hypergraph("largest_component2.csv")

    weights = compute_weights(H)
    hyperdegrees = compute_hyperdegrees(weights)

    max_hdeg = max(hyperdegrees.values())
    # 如果需要随机选择最大超度节点，可取消下一行注释并注释固定起点
    start_node = random.choice([n for n, d in hyperdegrees.items() if d == max_hdeg])
    #事实上，最大超度节点有两个，分别是"Elena De Momi"和"Jin U. Kang"
    print(f"The node with the maximum hyperdegree is '{start_node}' with a hyperdegree of {max_hdeg}.")

    restart_probs = [i * 0.00002 for i in range(51)]
    average_steps_by_restart_prob = {}

    for rp in restart_probs:
        steps_list = [perform_restart_random_walk(start_node, H, weights, restart_prob=rp) for _ in range(50)]
        avg_steps = sum(steps_list) / len(steps_list)
        average_steps_by_restart_prob[rp] = avg_steps
        print(f"Average number of steps with restart_prob={rp}: {avg_steps}")

    print("\nFinal Results:")
    for rp, avg in average_steps_by_restart_prob.items():
        print(f"Restart Probability: {rp:.5f}, Average Steps: {avg}")

    baseline_steps = average_steps_by_restart_prob[0.0]

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.scatter(average_steps_by_restart_prob.keys(),
                average_steps_by_restart_prob.values(),
                color="blue")
    plt.axhline(y=baseline_steps, color="red", linestyle="--",
                label=f"Baseline (Reset Rate=0): {baseline_steps:.2f}")
    plt.xlabel("Reset Rate", fontsize=20, fontweight="bold")
    plt.ylabel("Average Steps", fontsize=20, fontweight="bold")
    plt.legend(fontsize=18)
    plt.tick_params(axis="both", which="major", labelsize=18)
    plt.tight_layout()
    plt.savefig("5_4_new.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
