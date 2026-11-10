import numpy as np
import pandas as pd
import optuna
from sklearn.linear_model import LinearRegression

data = pd.read_csv("https://raw.githubusercontent.com/SanatRao13/engr100-finalproj/main/ENGR100Lab3Data.csv")
calib_data = data.drop(columns=['Abs1', 'Abs2', 'Abs3', 'AbsSD', 'Dye_Conc_pct'])


model = LinearRegression()

def r2(preds, y):
    y = np.squeeze(y)
    preds = np.squeeze(preds)
    return 1 - (np.sum((y - preds) ** 2) / np.sum((np.mean(y) - y) ** 2))

def objective(trial):
    global model
    pool = list(range(len(calib_data)))
    
    subset = []
    for i in range(8):
        idx = trial.suggest_categorical(f'row_{i}', pool)
        subset.append(idx)
        
    subset = list(set(subset))
    if len(subset) != 8:
        return float('-inf')
        
    X_train = calib_data.iloc[subset][['Dye_Conc_mgml']]
    y_train = calib_data.iloc[subset]['AvgAbs']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    preds = model.predict(X_train)
    r2_score = float(r2(preds, y_train))
    
    return r2_score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=1000)


print("--- Optimization Results ---")
print(f"Best R² Score: {study.best_value:.4f}")

best_params = study.best_params
best_subset = sorted([idx for idx in best_params.values()])
print("Best Selected Row Indices:", best_subset)
print("\nSelected Data Points:")
print(calib_data.iloc[best_subset][['Dye_Conc_mgml', 'AvgAbs']])


a = float(model.coef_)
b = float(model.intercept_)

import matplotlib.pyplot as plt

dat = calib_data.iloc[best_subset]
sds = data.iloc[best_subset]['AbsSD']
plt.scatter(dat['Dye_Conc_mgml'],dat['AvgAbs'])
plt.errorbar(dat['Dye_Conc_mgml'],dat['AvgAbs'],yerr=sds, fmt='bo', capsize=5, label='Average Absorbance at 530nm ± 1 SD')
plt.plot(pd.DataFrame(dat['Dye_Conc_mgml']),model.predict(pd.DataFrame(dat['Dye_Conc_mgml'])),'k-', label=f'{a:.5f}x+{b:.5f}')

plt.xlabel('Dye Concentration (mg/mL)')
plt.ylabel('Average Absorbance (au)')
plt.title('Average Absorbance at 530nm (au) vs Dye Concentration (mg/mL)')
plt.grid()

plt.xlim(0,0.30)
plt.ylim(0,3.0)
plt.legend()
plt.show()
