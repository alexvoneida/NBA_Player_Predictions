import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import train_test_split

#prepare data
df1 = pd.read_parquet('final_database_2024-25.parquet')
df2 = pd.read_parquet('final_database_2023-24.parquet')
df1 = df1.dropna()
df2 = df2.dropna()

place_df = df1.sort_values('GAME_DATE')
place_df2 = df2.sort_values('GAME_DATE')
split_index1 = int(len(place_df) * 0.6)

train_df = place_df.iloc[:split_index1]
train_df = pd.concat([train_df, place_df2])
test_df = place_df.iloc[split_index1:]

features = ['MIN_last5', 'PTS_last5', 'REB_last5', 'AST_last5', 'FG_PCT_last5', 'USAGE_last5', 'IS_HOME', 'DAYS_REST', 'PLUS_MINUS_last5', 'offensiveRating_last5', 'defensiveRating_last5', 'pace_last5', 'OPP_offensiveRating_last5', 'OPP_defensiveRating_last5', 'OPP_pace_last5']
targets = ['PTS', 'AST', 'REB', 'FG_PCT']

X_train = train_df[features].values.astype(np.float32)
y_train = train_df[targets]
X_test = test_df[features].values.astype(np.float32)
y_test = test_df[targets]

# Dictionary to store results
models = {}
results = {}
best_params_dict = {}

# Loop over each target
for stat in targets:
    print(f"\nOptimizing XGBoost model for {stat}...")

    # Split training data into training and validation for Optuna
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train[stat], test_size=0.2, random_state=42
    )

    # Define objective function for this stat
    def objective(trial):
        param = {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
            'random_state': 42
        }

        model = xgb.XGBRegressor(**param)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        return mae

    # Create Optuna study and run optimization
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50, show_progress_bar=True)

    # Train final model on full training data using best params
    best_params = study.best_params
    best_model = xgb.XGBRegressor(
        **best_params,
        objective='reg:squarederror',
        tree_method='hist'
    )
    best_model.fit(X_train, y_train[stat])
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test[stat], y_pred)

    # Store results
    models[stat] = best_model
    results[stat] = mae
    best_params_dict[stat] = best_params

    print(f"{stat} MAE on test set: {mae:.3f}")
    print(f"{stat} best hyperparameters: {best_params}")

for i in results:
    print(f"{i} MAE on test: {results[i]}")
'''
for stat, model in models.items():
    xgb.plot_importance(model, max_num_features=10)
    plt.title(f"Top Features for {stat}")
    plt.show()
'''

'''
PTS MAE on test: 4.8050642013549805
AST MAE on test: 1.4109216928482056
REB MAE on test: 2.0608582496643066
FG_PCT MAE on test: 0.18372800050938815
'''