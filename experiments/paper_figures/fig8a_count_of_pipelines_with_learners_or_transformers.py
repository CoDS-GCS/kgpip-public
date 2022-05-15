import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# pips = ['sklearn.ElasticNet','sklearn.make_pipeline','sklearn.LinearSVC','sklearn.cross_val_predict','sklearn.MiniBatchKMeans','sklearn.BaggingRegressor','sklearn.Birch','sklearn.MLPRegressor','sklearn.MeanShift','sklearn.SpectralClustering','sklearn.DBSCAN','sklearn.AffinityPropagation','sklearn.RandomizedSearchCV','sklearn.GaussianMixture','sklearn.Lasso','sklearn.MLPClassifier','sklearn.AdaBoostClassifier','sklearn.Ridge','sklearn.GradientBoostingRegressor','sklearn.SVR','sklearn.GradientBoostingClassifier','sklearn.MultinomialNB','sklearn.GaussianNB','sklearn.Pipeline','xgboost.XGBRegressor','sklearn.DecisionTreeRegressor','sklearn.SVC','sklearn.KMeans','sklearn.GridSearchCV','xgboost.XGBClassifier','sklearn.KNeighborsClassifier','sklearn.DecisionTreeClassifier','sklearn.RandomForestRegressor','sklearn.LogisticRegression','sklearn.RandomForestClassifier','sklearn.LinearRegression','xgboost.train']
# counts = [10,10,11,11,16,18,18,18,18,18,19,21,22,25,26,26,27,30,34,35,37,38,50,55,57,58,73,74,78,87,93,103,142,182,227,250,432]
pips = ['sklearn.AffinityPropagation','sklearn.RandomizedSearchCV','sklearn.GaussianMixture','sklearn.Lasso','sklearn.MLPClassifier','sklearn.AdaBoostClassifier','sklearn.Ridge','sklearn.GradientBoostingRegressor','sklearn.SVR','sklearn.GradientBoostingClassifier','sklearn.MultinomialNB','sklearn.GaussianNB','sklearn.Pipeline','xgboost.XGBRegressor','sklearn.DecisionTreeRegressor','sklearn.SVC','sklearn.KMeans','sklearn.GridSearchCV','xgboost.XGBClassifier','sklearn.KNeighborsClassifier','sklearn.DecisionTreeClassifier','sklearn.RandomForestRegressor','sklearn.LogisticRegression','sklearn.RandomForestClassifier','sklearn.LinearRegression','xgboost.train']
counts = [21,22,25,26,26,27,30,34,35,37,38,50,55,57,58,73,74,78,87,93,103,142,182,227,250,432]

pips.reverse()
counts.reverse()

colors = ['cornflowerblue' if i.startswith('sklearn') else 'orange' for i in pips]
pips = [i.split('.')[1] for i in pips]



# plt.style.use('seaborn-white')
plt.rcParams["font.size"] = 17
plt.rcParams['axes.axisbelow'] = True


fig, ax = plt.subplots(1,1, figsize=(12, 8))
fig.patch.set_facecolor('white')



ax.barh(np.arange(len(counts)), counts, align='center', color=colors)
ax.set_yticks(np.arange(len(counts)), pips)
ax.set_xticks(np.arange(0, 450, 25))
ax.set_xticklabels(np.arange(0, 450, 25), rotation=90)
ax.set_ylabel('Learner / Transformer', fontsize=25)
ax.set_xlabel('Pipeline Count', fontsize=30)

labels = ['Sklearn Call', 'XGBoost Call']
handles = [plt.Rectangle((0,0),1,1, color=c) for c in ['cornflowerblue', 'orange']]
ax.legend(handles, labels, fontsize=18)
ax.grid(linestyle = '--', linewidth = 1, axis='x')


plt.tight_layout()
plt.savefig('figure_counts_of_pipelines_with_learners_or_transformers.pdf', bbox_inches='tight', pad_inches=0)
plt.show()