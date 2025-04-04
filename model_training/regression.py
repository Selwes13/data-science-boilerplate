#%%
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error, mean_absolute_error

#%% load data
df = pd.read_csv("")

#%% features
xs = []
y = ''

X_train, X_test, y_train, y_test = train_test_split(
    df[xs], df[y], 
    test_size=0.33, random_state=42)

#%% Grid Search (Random Forest)

param_grid = {
    'n_estimators':[10, 25, 50, 100, 150],
    'max_depth':[1, 2, 4, 8, 12]
}
base_model = RandomForestRegressor()
grid_search = GridSearchCV(base_model, param_grid, scoring='r2')

grid_search.fit(X_train, y_train)

#%% Pull best parameters
df_results = pd.DataFrame(grid_search.cv_results_)
best_params = df_results.query("rank_test_score == 1")['params']

print(grid_search.best_estimator_)
best_model = grid_search.best_estimator_

#%% Train model with best parameters
#model = RandomForestRegressor(max_depth=4, random_state=827634)
model = RandomForestRegressor(
    max_depth=best_model.max_depth, 
    n_estimators=best_model.n_estimators
    )

# Fit the model
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate metrics
scores = {
    'r2':r2_score(y_test, y_pred),
    'MAE':mean_absolute_error(y_test, y_pred),
    'MSE':mean_absolute_error(y_test, y_pred),
    'R-MSE':root_mean_squared_error(y_test, y_pred) 
}

for score_lbl in scores:
    print(f"{score_lbl}: {scores[score_lbl]}")

#%% Predicted v Actual
max_val = max([max(y_test), max(y_pred)])

plt.figure(figsize=(4.5,4))
plt.scatter(y_test, y_pred, marker='x')
plt.plot([0,max_val], [0,max_val], color='k', linestyle='--')
plt.ylabel("Predicted")
plt.xlabel("Actual")

plt.ylim(-5, max_val)
plt.xlim(-5, max_val)

plt.show()