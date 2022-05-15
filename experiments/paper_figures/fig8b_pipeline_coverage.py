
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def rotate(l, n):
    return l[-n:] + l[:-n]

pips = ['random_forest', 'extra_tree', 'sgd', 'gradient_boosting', 'xgboost', 'k_nearest_neighbors', 'decision_tree', 'adaboost', 'pca', 'libsvm_svc', 'libsvm_svr', 'Other']
counts = [500, 388, 260, 254, 225, 147, 142, 114, 74, 45, 40, 126]
# colors = ["#1255d3", "#964550", "#1eacc9", "#cc224d", "#1fc468", "#fa41c7", "#1e6c3c", "#dc8bfe", "#4d5e87", "#9e99c1","#85b973", "#de8a2c"]
colors = ['#8cc63f','#8b9fc0', '#87cefa', '#eec4ab',  '#ff7bac','#adaaaa', '#ffd700', '#e14f4f', '#3d6c87', '#1255d3', '#d24726', '#f0ad4e']

# pips = rotate(pips, 4)
# counts = rotate(counts, 4)
# colors = rotate(colors, 4)

plt.style.use('seaborn-muted')
plt.rcParams["font.size"] = 13


fig, ax = plt.subplots(1,1, figsize=(12, 8))
fig.patch.set_facecolor('white')
angle = 35
slices, labels, _ = ax.pie(counts, labels = pips, colors=colors, startangle=angle, autopct='%1.0f%%', pctdistance=0.85, labeldistance=1.05, textprops={'fontsize': 20, 'color':'k'}, wedgeprops={"edgecolor":"white",'linewidth': 2, 'linestyle': 'dashed', 'antialiased': True})
center = slices[0].center
r = slices[0].r
circle = matplotlib.patches.Circle(center, r, fill=False, edgecolor="black", linewidth=2.5)
ax.add_patch(circle)
# plt.title('Count of All Choices', fontsize=30, weight='bold')
plt.savefig('figure_pipeline_coverage.pdf', bbox_inches='tight', pad_inches=0)
plt.show() 
