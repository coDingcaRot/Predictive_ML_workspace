# --- Imports ---
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, make_scorer

# --- 1. Load Data ---
print('Loading Data...')
train_df = pd.read_csv("./data/train.csv")
train_df["HighRisk"] = train_df["HighRisk"].map({"Yes": 1, "No": 0})

# --- 2. Choose Top Features (from EDA) ---
selected_features = [
    "ECigaretteUsage",
    "MentalHealthDays",
    "HIVTesting",
    "HadDepressiveDisorder",
    "DifficultyConcentrating",
]

X = train_df[selected_features]
y = train_df["HighRisk"]

# --- 3. Identify column types ---
print('\nSeparating numerical and categorical columns...')
numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()


# 3. Preprocessing
print('\nCreating Preprocessors...')
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_cols),
    ("cat", cat_pipeline, categorical_cols)
])

# 4. Define models
print('\nDefining Models...')
models = {
    "rf": RandomForestClassifier(n_estimators=100),
    "gb": GradientBoostingClassifier(n_estimators=100),
    "lr": LogisticRegression(max_iter=500),
    "ann": MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', learning_rate_init=0.01,
                          max_iter=300)
}

# 5. Evaluate models with KFold
print('\nKfold Evaluations...')
kf = KFold(n_splits=5, shuffle=True)
f1 = make_scorer(f1_score)

trained_models = {}
if not os.path.exists("BinaryFolder"):
    os.makedirs("BinaryFolder")

for name, model in models.items():
    print(f'Training {name}')
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", model)
    ])
    score = cross_val_score(pipe, X, y, cv=kf, scoring=f1)
    print(f"{name} F1 avg: {score.mean():.4f}, std: {score.std():.4f}")
    pipe.fit(X, y)
    trained_models[name] = pipe
    joblib.dump(pipe, f"./BinaryFolder/{name}_model.pkl")

# 6. Stacking model
print('\nStacking Model Creation')
estimators = [(k, trained_models[k]["clf"]) for k in ["rf", "gb"]]
stack_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=500),
    passthrough=True
)

# Train full stacking pipeline
stack_pipe = Pipeline([
    ("pre", preprocessor),
    ("clf", stack_model)
])
stack_pipe.fit(X, y)
joblib.dump(stack_pipe, "./BinaryFolder/stacked_model.pkl")
print("Stacked model saved.")

# Save preprocessors separately (optional)
joblib.dump(preprocessor, "./BinaryFolder/preprocessor.pkl")

# 7.
print('\nEvaluation metrics...')
# Load preprocessor
preprocessor = joblib.load("BinaryFolder/preprocessor.pkl")

# Model paths
model_paths = {
    "Random Forest": "./BinaryFolder/rf_model.pkl",
    "Gradient Boosting": "./BinaryFolder/gb_model.pkl",
    "Logistic Regression": "./BinaryFolder/lr_model.pkl",
    "ANN": "./BinaryFolder/ann_model.pkl",
    "Stacked Model": "./BinaryFolder/stacked_model.pkl"
}

# KFold setup
kf = KFold(n_splits=5, shuffle=True)
f1 = make_scorer(f1_score)

# Evaluate models
for name, path in model_paths.items():
    print(f'\nEvaluating Model... {name}')
    model = joblib.load(path)
    score = cross_val_score(model, X, y, cv=kf, scoring=f1)
    print(f"{name}: F1 avg = {score.mean():.4f}, std = {score.std():.4f}")