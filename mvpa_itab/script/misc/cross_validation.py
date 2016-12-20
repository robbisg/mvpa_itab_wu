


for i in np.arange(2,11):
    this_scores = cross_val_score(svc, X, y, cv=StratifiedKFold(n_folds=i), scoring='accuracy',n_jobs=1)
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
