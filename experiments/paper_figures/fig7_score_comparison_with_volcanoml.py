import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('kgpip_results_vs_volcano_44_datasets.csv')

# filter datasets with equal scores

df['diff'] = df['KGpipFLAML'] - df['VolcanoWE']

df = df[(df['diff'].abs().round(2) > 0.01)]
df['color'] = df['diff'].apply(lambda x: 'mediumseagreen' if x > 0 else 'indianred')
df = df.sort_values('diff')

print(len(df), 'Datasets')

reg = df[df['Task'] == 'regression']
cls = df[df['Task'] != 'regression']

plt.style.use('seaborn-white')
plt.rcParams["font.size"] = 18

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,7))
fig.patch.set_facecolor('white')
reg_ids = reg['ID'].astype(str).tolist()
reg_diffs = reg['diff'].tolist()
cls_ids = cls['ID'].astype(str).tolist()
cls_diffs = cls['diff'].tolist()


ax1.bar(np.arange(len(cls_diffs)), cls_diffs, align='center', color=cls.color.tolist())
ax1.set_xticks(np.arange(len(cls_diffs)), cls_ids, rotation=90)
ax1.set_yticks(np.arange(-0.05, 1, 0.05))
ax1.set_ylabel('F1-Score Absolute Difference', fontsize=25)
ax1.set_xlabel('Classification Datasets', fontsize=30)
ax1.grid(linestyle = '--', linewidth = 1, axis='y')


ax2.bar(np.arange(len(reg_diffs)), reg_diffs, align='center', color=reg.color.tolist())
ax2.set_xticks(np.arange(len(reg_diffs)), reg_ids, rotation=90)
ax2.set_yticks(np.arange(0, max(reg_diffs)+0.05, 0.05))
ax2.set_ylabel('R² Absolute Difference', fontsize=25)
ax2.set_xlabel('Regression Datasets', fontsize=30)
ax2.grid(linestyle = '--', linewidth = 1, axis='y')
ax2.set_facecolor('white')


plt.tight_layout()
plt.savefig('figure_kgpip_vs_volcano.pdf', bbox_inches='tight', pad_inches=0)
plt.show()
