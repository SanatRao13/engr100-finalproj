import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LinearRegression
import optuna.visualization as vis
import matplotlib.pyplot as plt

# importing data
data = pd.read_csv("https://raw.githubusercontent.com/SanatRao13/engr100-finalproj/main/ENGR100Lab3Data.csv")
calib_data = data.drop(columns=['Abs1', 'Abs2', 'Abs3', 'AbsSD', 'Dye_Conc_pct'])

#initialize global LR model
model = LinearRegression()

#defining R2 (using manual definition to avoid R2>=0 restriction)
def r2(preds, y):
    y = np.squeeze(y)
    preds = np.squeeze(preds)
    return 1 - (np.sum((y - preds) ** 2) / np.sum((np.mean(y) - y) ** 2))

def objective(trial):
    global model
    pool = list(range(len(calib_data))) # pool of available indices (all indices)
    
    subset = []

    # using BO-TPE to suggest 8 points for regression
    for i in range(8):
        idx = trial.suggest_categorical(f'row_{i}', pool)
        subset.append(idx)

    # removes duplicates and punishes duplicate rows
    subset = list(set(subset))
    if len(subset) != 8:
        return float('0')
        
    X_t = calib_data.iloc[subset][['Dye_Conc_mgml']]
    y_t = calib_data.iloc[subset]['AvgAbs']

    # generate linear regression & get R2 score
    model = LinearRegression()
    model.fit(X_t, y_t)
    preds = model.predict(X_train)
    r2_score = float(r2(preds, y_train))
    
    return r2_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1500)

# Optimization receipt
print("--- Optimization Results ---")
print(f"Best R² Score: {study.best_value:.4f}")

best_params = study.best_params
best_subset = sorted([idx for idx in best_params.values()])
print("Best Selected Row Indices:", best_subset)
print("\nSelected Data Points:")
print(calib_data.iloc[best_subset][['Dye_Conc_mgml', 'AvgAbs']])

# generate optimization graphic
vis.plot_optimization_history(study)

# get LR params
a = float(model.coef_)
b = float(model.intercept_)

# plot selected points, regression, and error bars
dat = calib_data.iloc[best_subset]
sds = data.iloc[best_subset]['AbsSD']
plt.scatter(dat['Dye_Conc_mgml'],dat['AvgAbs'])
plt.errorbar(dat['Dye_Conc_mgml'],dat['AvgAbs'],yerr=sds, fmt='bo', capsize=5, label='Average Absorbance at 530nm ± 1 SD')
plt.plot(pd.DataFrame(dat['Dye_Conc_mgml']),model.predict(pd.DataFrame(dat['Dye_Conc_mgml'])),'k-', label=f'{a:.5f}x+{b:.5f}')

# format axes, and figure settigns to optimize detailed layout
plt.xlabel('Dye Concentration (mg/mL)')
plt.ylabel('Average Absorbance (au)')
plt.title('Average Absorbance at 530nm (au) vs Dye Concentration (mg/mL)')
plt.grid()
plt.xlim(0,0.30)
plt.ylim(0,3.0)
plt.legend()
plt.show()
