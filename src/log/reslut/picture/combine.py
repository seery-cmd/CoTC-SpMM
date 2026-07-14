import matplotlib.pyplot as plt

# 数据从打印输出中提取
groups = ['Extended Magnitude', 'Magnitude', 'Random', 'Variational Dropout']
q_values = [26.0878, 40.9916, 39.9076, 37.9108]
r_values = [41.4720, 39.9550, 37.6794, 40.3342]
s_values = [43.9016, 57.3788, 55.6405, 53.7031]

# 计算第一个柱子的总和（Q + R）
first_bar_totals = [q + r for q, r in zip(q_values, r_values)]

# 设置柱子的位置
x = range(len(groups))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 4))
color_q = '#fabb9e'   # 浅橙
color_q_dark = '#d4883e'  # 深橙
color_r = '#fbdd9c'   # 浅黄
color_r_dark = '#d4aa3e'  # 深黄
color_s = '#94d8da'   # 浅青
color_s_dark = '#4ca8aa'  # 深青

# 绘制堆叠柱子，每个柱子使用自己底色的深色作为 edgecolor/hatch 颜色
ax.bar(x, q_values, width, label='TC kernel', color=color_q,
       edgecolor=color_q_dark, linewidth=0.8, zorder=1, hatch='/')
ax.bar(x, r_values, width, bottom=q_values, label='CUDA kernel', color=color_r,
       edgecolor=color_r_dark, linewidth=0.8, zorder=1, hatch='\\')
ax.bar([i + width for i in x], s_values, width, label='CoTC-SpMM', color=color_s,
       edgecolor=color_s_dark, linewidth=0.8, zorder=1)


# 自定义图例：横着在图外上部
# ax.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3, fontsize=12)
ax.set_ylabel('Kernel time(us)', fontsize=20)
ax.set_xticks([i + width/2 for i in x])
ax.set_xticklabels(groups, fontsize=14)

ax.set_ylim(0, 100)                # 范围 0~80
ax.set_yticks([0, 20, 40, 60, 80, 100])  # 间隔 20

# 添加浅色水平网格线，设置zorder低于柱子
ax.grid(axis='y', alpha=0.28, color='lightgray', zorder=0)


# 自定义图例：横着在图外上部，比标题还往上
ax.legend(title=None, loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=3, fontsize=14)


plt.tight_layout()

# 保存图为PDF
plt.savefig('scaled_means_comparison3.pdf', format='pdf', bbox_inches='tight')

plt.show()