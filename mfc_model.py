import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load the expanded, grounded dataset
data = pd.read_csv("mfc_data_v2.csv")

# 2. Feature Engineering Function
def engineer_features(df):
    # BOD/COD ratio (Bio-marker)
    df['BOD_COD_Ratio'] = df['BOD_in'] / (df['COD_in'] + 1e-6)
    # Organic Load (Total amount of fuel)
    df['Organic_Load'] = (df['COD_in'] * df['Volume']) / 1000.0
    # pH Deviation from optimal (neutrality is often best)
    df['pH_Dev'] = abs(df['pH_in'] - 7.2)
    # Encode Category (Real vs Synthetic)
    # Handle the fact that during training we have both, but during prediction in app, 
    # we might want to "force" the model to behave like it's predicting 'Real' data.
    df['Is_Real'] = (df['Category'] == 'Real').astype(int)
    return df

data = engineer_features(data)

# 3. Define Input Features (X)
# Including Is_Real as a feature allows comparing Real vs Synthetic bias
feature_cols = ['WW_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in', 
                'BOD_COD_Ratio', 'Organic_Load', 'pH_Dev', 'Is_Real']
X = data[feature_cols]

# 4. Define Output Targets (y)
target_cols = ['Voltage', 'Power_Density', 'Coulombic_Efficiency', 
               'COD_out', 'BOD_out', 'pH_out']
y = data[target_cols]

# 5. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Train Individual Models with Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV

models = {}
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6],
    'min_samples_split': [2, 5, 10],
    'subsample': [0.8, 0.9, 1.0]
}

for target in target_cols:
    print(f"Tuning and training specialized model for {target}...")
    base_model = GradientBoostingRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, 
        n_iter=15, cv=3, scoring='r2', n_jobs=-1, random_state=42
    )
    random_search.fit(X_train_scaled, y_train[target])
    models[target] = random_search.best_estimator_
    print(f"  Best Params for {target}: {random_search.best_params_}")

# 8. Evaluation Wrapper (Updated to handle Is_Real)
class MFCModelWrapper:
    def __init__(self, models, scaler, feature_cols, target_cols):
        self.models = models
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        
    def predict(self, X_raw, predict_real=True):
        # Handle both single input and batch
        if isinstance(X_raw, np.ndarray):
            X_df = pd.DataFrame(X_raw, columns=['WW_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in'])
        else:
            X_df = X_raw.copy()
            
        # We add 'Category' as 'Real' if predict_real=True, else 'Synthetic'
        X_df['Category'] = 'Real' if predict_real else 'Synthetic'
        X_eng = engineer_features(X_df)
        X_scaled = self.scaler.transform(X_eng[self.feature_cols])
        
        preds = []
        for target in self.target_cols:
            preds.append(self.models[target].predict(X_scaled))
            
        return np.column_stack(preds)

final_model = MFCModelWrapper(models, scaler, feature_cols, target_cols)

# 9. Evaluate
# Using the test set with its original category
X_test_raw = X_test.copy()
X_test_raw_inputs = X_test_raw[['WW_Type', 'Volume', 'COD_in', 'BOD_in', 'pH_in']]

# Prediction based on actual categories in test set
X_test_eng = X_test.copy() # Already has Is_Real
X_test_scaled_direct = scaler.transform(X_test_eng[feature_cols])
predictions_direct = []
for target in target_cols:
    predictions_direct.append(models[target].predict(X_test_scaled_direct))
predictions_direct = np.column_stack(predictions_direct)

overall_r2 = r2_score(y_test, predictions_direct)
print("-" * 30)
print(f"Fine-Tuning Complete (Dual Dataset).")
print(f"Overall R2 Score: {overall_r2:.4f}")

# Individual Target Scores
print("\nAccuracy per target (R2 Score):")
for i, target in enumerate(target_cols):
    score = r2_score(y_test.iloc[:, i], predictions_direct[:, i])
    print(f"- {target}: {score:.4f}")

# 10. Save Model as a Dictionary
model_assets = {
    'models': models,
    'scaler': scaler,
    'feature_cols': feature_cols,
    'target_cols': target_cols
}

joblib.dump(model_assets, "mfc_trained_model.pkl")
print("\nFinal Optimized Model assets saved as a dictionary (Including Real/Synthetic Category).")
