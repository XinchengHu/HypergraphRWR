# 计算超图稳定分布和对应团图的稳定分布，以此进行节点排序

import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import combinations
from matplotlib import rcParams

# -------------------- 全局字体配置（保持原功能） --------------------
rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,
    "mathtext.fontset": "stix",
    "font.serif": ["SimSun"],
    "axes.unicode_minus": False,
    "font.weight": "bold"
})

# -------------------- 读取与构建超图 --------------------
df = pd.read_csv('largest_component.csv', encoding='ISO-8859-1')

hypergraph = defaultdict(set)          # 论文标题 -> 节点索引集合
nodes = {}                              # 作者名 -> 节点索引
node_degree_count = defaultdict(int)    # 投影图中节点度（每篇论文为每位作者贡献 m-1）

for _, row in df.iterrows():
    paper = row['Title']
    authors = [a.strip() for a in row['Authors'].split(',') if a.strip()]
    if not authors:
        continue
    # 建立作者索引 & 论文作者索引列表
    author_ids = [nodes.setdefault(a, len(nodes)) for a in authors]
    m = len(author_ids)

    # 投影图度数：每位作者 + (m-1)
    for uid in author_ids:
        node_degree_count[uid] += (m - 1)

    # 超图：论文 -> 作者集合
    hypergraph[paper].update(author_ids)

print(f"Number of hyperedges: {len(hypergraph)}")
print(f"Number of nodes: {len(nodes)}")

# -------------------- 计算节点间权重（团图加权）与超度 --------------------
weights = defaultdict(lambda: defaultdict(int))  # i -> {j: w_ij}
for authors in hypergraph.values():
    s = len(authors)
    if s < 2:
        continue
    w = s - 1
    # 原实现对 i!=j 双向累加；此处用两两组合并对称累加，等价
    for i, j in combinations(authors, 2):
        weights[i][j] += w
        weights[j][i] += w

node_hyperdegrees = {i: sum(neigh.values()) for i, neigh in weights.items()}
max_hyperdegree = max(node_hyperdegrees.values())
print(f"max_hyperdegree is {max_hyperdegree}")

# 稳定分布 p/maxp
node_stable_distribution = {i: h / max_hyperdegree for i, h in node_hyperdegrees.items()}

# -------------------- 投影稳定分布（q/maxq） --------------------
max_degree = max(node_degree_count.values())
print(f"max_degree is {max_degree}")
node_projected_stable_distribution = {i: d / max_degree for i, d in node_degree_count.items()}

# -------------------- 画图小工具（避免重复） --------------------
def scatter_and_save(x, y, *, color, xlabel, ylabel, out, s=10, xlim=None, ylim=(0, 1), tick=20):
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color=color, s=s)
    if xlim: plt.xlim(*xlim)
    if ylim: plt.ylim(*ylim)
    plt.xlabel(xlabel, fontsize=20, fontweight='bold')
    plt.ylabel(ylabel, fontsize=20, fontweight='bold')
    plt.tick_params(axis='both', which='major', labelsize=tick)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.show()

# -------------------- 图1：所有节点 p/maxp --------------------
nodes_list_1 = list(node_stable_distribution.keys())
p_vals = [node_stable_distribution[i] for i in nodes_list_1]
scatter_and_save(
    nodes_list_1, p_vals,
    color='b',
    xlabel='Node Index',
    ylabel=r'$P_i^{\infty} / \max_j P_j^{\infty}$',
    out='5_1.png'
)

# -------------------- 图2：所有节点 q/maxq --------------------
nodes_list_2 = list(node_projected_stable_distribution.keys())
q_vals = [node_projected_stable_distribution[i] for i in nodes_list_2]
scatter_and_save(
    nodes_list_2, q_vals,
    color='r',
    xlabel='Node Index',
    ylabel=r'$Q_i^{\infty} / \max_j Q_j^{\infty}$',
    out='5_2.png'
)

# -------------------- 图3：q 对 p（含对角线） --------------------
plt.figure(figsize=(10, 6))
plt.scatter(q_vals, p_vals, color='g', s=10)
plt.xlim(0, 1); plt.ylim(0, 1)
plt.plot([0, 1], [0, 1], color='black', linestyle='-', linewidth=1)
plt.xlabel(r'$Q_i^{\infty} / \max_j Q_j^{\infty}$', fontsize=20, fontweight='bold')
plt.ylabel(r'$P_i^{\infty} / \max_j P_j^{\infty}$', fontsize=20, fontweight='bold')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.grid(True)
plt.tight_layout()
plt.savefig('5_3.png', dpi=300)
plt.show()
