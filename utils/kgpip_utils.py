supported_autosklearn_classifiers = {
    'XGBClassifier': 'gradient_boosting',  # XGBoost algorithm is implemented in sklearn
    'GradientBoostingClassifier': 'gradient_boosting',
    'AdaBoostClassifier': 'adaboost',
    'RandomForestClassifier': 'random_forest',
    'SGDClassifier': 'sgd',
    'LogisticRegression': 'sgd',
    'RidgeClassifier': 'sgd',
    'ExtraTreesClassifier': 'extra_trees',
    'KNeighborsClassifier': 'k_nearest_neighbors',
    'DecisionTreeClassifier': 'decision_tree',
    'MultinomialNB': 'multinomial_nb',
    'BernoulliNB': 'bernoulli_nb',
    'LinearSVC': 'liblinear_svc',
    'SVC': 'libsvm_svc',
    'MLPClassifier': 'mlp',
    'GaussianNB': 'gaussian_nb',
    'LinearDiscriminantAnalysis': 'lda',
    'PassiveAggressiveClassifier': 'passive_aggressive',
    'QuadraticDiscriminantAnalysis': 'qda'
}
supported_autosklearn_regressors = {
    'XGBRegressor': 'gradient_boosting',  # XGBoost algorithm is implemented in sklearn
    'GradientBoostingRegressor': 'gradient_boosting',
    'AdaBoostRegressor': 'adaboost',
    'RandomForestRegressor': 'random_forest',
    'SGDRegressor': 'sgd',
    'LinearRegression': 'sgd',
    'Ridge': 'sgd',
    'Lasso': 'sgd',
    'KNeighborsRegressor': 'k_nearest_neighbors',
    'ExtraTreesRegressor': 'extra_trees',
    'DecisionTreeRegressor': 'decision_tree',
    'LinearSVR': 'liblinear_svr',
    'SVR': 'libsvm_svr',
    'MLPRegressor': 'mlp',
    'GaussianProcessRegressor': 'gaussian_process',
    'ARDRegression': 'ard_regression',
}
supported_autosklearn_preprocessors = {
    'FastICA': 'fast_ica',
    'FeatureAgglomeration': 'feature_agglomeration',
    'KernelPCA': 'kernel_pca',
    'RBFSampler': 'kitchen_sinks',
    'Nystroem': 'nystroem_sampler',
    'PCA': 'pca',
    'PolynomialFeatures': 'polynomial',
    'RandomTreesEmbedding': 'random_trees_embedding',
    'SelectPercentile': 'select_percentile',
    'TruncatedSVD': 'truncatedSVD'
}
supported_flaml_classifiers = {
    'XGBClassifier': 'xgboost',
    'GradientBoostingClassifier': 'xgboost',
    'RandomForestClassifier': 'rf',
    'LGBMClassifier': 'lgbm',
    'CatBoostClassifier': 'catboost',
    'ExtraTreesClassifier': 'extra_tree',
    # 'LogisticRegression': ['lrl2', 'lrl1'],    # Use both L1 and L2 regularizations for logistic regression.
    # 'KNeighborsClassifier': 'kneighbor',
}
supported_flaml_regressors = {
    'XGBRegressor': 'xgboost',
    'GradientBoostingRegressor': 'xgboost',
    'RandomForestRegressor': 'rf',
    'LGBMRegressor': 'lgbm',
    'CatBoostRegressor': 'catboost',
    'ExtraTreesClassifier': 'extra_tree',
    # 'KNeighborsRegressor': 'kneighbor',
}
supported_flaml_preprocessors = {}

def is_text_column(df_column):
    """
    Check if the dataframe column is a text (as opposed to short strings).
    Simple heuristic: check if it has 20+ unique values and 30%+ of the column contains 2+ space characters.
    TODO: this will not work with timestamp columns that contain spaces.
    """
    # to speed up the search in case of string column, check how many unique values
    if df_column.dtype != object or len(df_column.unique()) < 20:
        return False

    num_text_rows = 0
    for value in df_column.values.tolist():
        if not isinstance(value, str):
            continue
        space_count = 0
        for character in value:
            if character == ' ':
                space_count += 1
            if space_count > 1:
                num_text_rows += 1
                break
        if num_text_rows > 0.3 * len(df_column):
            return True
    return False
