# Utilities
from os.path import split
from matplotlib.pyplot import plot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MaxAbsScaler, MinMaxScaler
import seaborn as sns

# Classifiers or Regressors + Utilities
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from mlxtend.classifier      import EnsembleVoteClassifier
from xgboost                 import XGBClassifier, plot_importance
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split, KFold, cross_validate  # test train split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

######################################################
########## Initial Loading values and setup ##########
######################################################

obesity_df = pd.read_csv('Obesity_dataset.csv')
obesity_df = obesity_df.rename(columns={"NObeyesdad": "obesity_status",
            "FAVC": "freq_high_calorie_food_eat",
            "FCVC": "daily_vegetable_consumption",
            "NCP": "nutrition_care_process",
            "CAEC": "food_between_meals",
            "CH2O": "daily_water_drinking",
            "SCC": "caloric_monitoring",
            "FAF": "daily_physical_activity",
            "TUE": "daily_electronic_usage",
            "CALC": "freq_alcohol_usage",
            "MTRANS": "transportation_method",})
order = ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I', 'Overweight_Level_II',
 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III']
sub_order = ['no', 'Sometimes', 'Frequently', 'Always'] # order used for
ordinal = ["food_between_meals", "freq_alcohol_usage", "obesity_status"]

########################################
########## Data Preprocessing ##########
########################################

def remove_duplicates(df):
    """
    Removes duplicates of a given dataframe
    :param df: as a pandas dataframe
    :return: a dataframe with duplicates removed
    """
    dataframe = df.copy()
    dataframe = dataframe.drop_duplicates()

    return dataframe
obesity_df = remove_duplicates(obesity_df)

def smart_encode_categorical(df, ordinal_mappings):
    """
    Encodes categorical columns in a DataFrame based on specified ordinal mappings.
    Nominal categorical columns are one-hot encoded with separate columns for each category.

    Args:
        df: The pandas DataFrame.
        ordinal_mappings: A dictionary where keys are ordinal column names and values are lists defining the order.

    Returns:
        A new DataFrame with categorical columns encoded.
    """
    df_encoded = df.copy()

    # One-hot encode nominal columns (all object columns not in ordinal_mappings)
    nominal_cols = [col for col in df_encoded.select_dtypes(include=['object']).columns if col not in ordinal_mappings]

    for col in nominal_cols:
        dummies = pd.get_dummies(df_encoded[col], prefix=col, dtype="float64")
        df_encoded = pd.concat([df_encoded, dummies], axis=1)  # Keep all dummy columns
        df_encoded.drop(columns=[col], inplace=True)  # Remove original categorical column

    # Apply ordinal encoding based on predefined mappings
    for col, order in ordinal_mappings.items():
        if col in df_encoded.columns:
            mapping = {category: idx for idx, category in enumerate(order)}
            df_encoded[col] = df_encoded[col].map(mapping).astype("float64")  # Ensure it's properly mapped

    return df_encoded
# Mapping of values to my ordinal
ordinal_mappings = {
    "obesity_status": ['Insufficient_Weight', 'Normal_Weight', 'Overweight_Level_I',
                       'Overweight_Level_II', 'Obesity_Type_I', 'Obesity_Type_II', 'Obesity_Type_III'],
    "food_between_meals": ['no', 'Sometimes', 'Frequently', 'Always'],
    "freq_alcohol_usage": ['no', 'Sometimes', 'Frequently', 'Always']
}
obesity_df_cleaned = smart_encode_categorical(obesity_df, ordinal_mappings) # cleaned obesity dataframe

#######################################
########## FEATURE SELECTION ##########
#######################################

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
    "Ridge Classifier": RidgeClassifier(),
    "SVC Classifier": SVC(kernel='linear', C=10),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=500), # takes a long time
    "Decision Tree Classifier": DecisionTreeClassifier(), #takes a long time
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XGB Classifier": XGBClassifier(),
}
def feature_selection(X, y, top_feature_num, classifiers, random_state=None):
    print(f"""\n
############################################
########## Feature Selection List ##########
############################################
""")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    all_importances = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)  # Train the classifier
        print(f"\n{name} Feature Importance's:")

        # Get feature importance or coefficients
        if hasattr(clf, 'feature_importances_'):
            importance = clf.feature_importances_
        elif hasattr(clf, 'coef_'):
            coefficient = clf.coef_ # Checks if coefficient dimension (ndim) is greater than 1.
            if coefficient.ndim > 1:  # Handle multi-class coefficients
                importance = np.mean(np.abs(coefficient), axis=0)  # Average across classes
            else:
                importance = np.abs(coefficient)  # Single class
            importance = importance / np.sum(importance)  # Normalize
        else:
            print(f"{name} does not have feature importances or coefficients.")
            continue

        # Create a DataFrame for feature importance
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importance})
        feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
        print(feature_importance_df.head(top_feature_num))  # Show top features
        all_importances.append(importance)

    # Calculate mean importance across all models
    mean_importance = np.mean(all_importances, axis=0)
    mean_feature_df = pd.DataFrame({'Feature': X.columns, 'Mean Importance': mean_importance})
    mean_feature_df = mean_feature_df.sort_values(by='Mean Importance', ascending=False)
    print("\nMean Feature Importance Between All Classifiers:")
    print(mean_feature_df.head(top_feature_num))

X = obesity_df_cleaned.copy()
scaler = StandardScaler() #Consistently the highest
# scaler = MaxAbsScaler()
# scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

y = X["obesity_status"]
del X["obesity_status"]
del X_scaled["obesity_status"]
feature_selection(X_scaled, y, 5, classifiers, None)
best_X = X_scaled[["Weight", "Height", "Gender_Female", "daily_vegetable_consumption", "Age"]] # Predictor set 1

#####################################
########## MODEL SELECTION ##########
#####################################

# Bagged model
def determine_bagged_model(X, y, classifier_list, random_state=None):
    from sklearn.base import clone

    print(f"""\n
#############################################
########## Bagging Model Selection ##########
#############################################""")

    # Split the data into train (60%), validation (20%), and test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

    for name, clf in classifier_list.items():
        print(f"\n********** Bagged {name}: **********")

        # Initialize and train the bagging classifier
        bagging_clf = BaggingClassifier(
            estimator=clone(clf),
            n_estimators=100,  # Number of base estimators
            max_samples=0.8,  # Percentage of samples to draw for each base estimator
            max_features=0.8,  # Percentage of features to draw for each base estimator
            random_state=random_state)
        bagging_clf.fit(X_train, y_train)

        # Validation Set Evaluation
        y_val_pred = bagging_clf.predict(X_val)
        print("\nValidation Set Evaluation:")
        print("Precision:", precision_score(y_val, y_val_pred))
        print("Recall:", recall_score(y_val, y_val_pred))
        print("F1-Score:", f1_score(y_val, y_val_pred))
        print(classification_report(y_val, y_val_pred))

        # Test Set Evaluation (Unseen Data)
        y_test_pred = bagging_clf.predict(X_test)
        print("\nTest Set Evaluation (Unseen Data):")
        print("Precision:", precision_score(y_test, y_test_pred))
        print("Recall:", recall_score(y_test, y_test_pred))
        print("F1-Score:", f1_score(y_test, y_test_pred))
        print(classification_report(y_test, y_test_pred))

        # K-Fold Cross Validation (on the full training set)
        scoring = {
            'accuracy': 'accuracy',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted',
            'f1_weighted': 'f1_weighted'
        }
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)
        cv_results = cross_validate(bagging_clf, X_train, y_train, cv=kfold, scoring=scoring)

        print("\nK-Fold Validation Results (Training Set):")
        print("Mean Accuracy:", round(cv_results['test_accuracy'].mean(), 4))
        print("Standard Deviation (Accuracy):", round(cv_results['test_accuracy'].std(), 4))
        print("Mean Precision:", round(cv_results['test_precision_weighted'].mean(), 4))
        print("Standard Deviation (Precision):", round(cv_results['test_precision_weighted'].std(), 4))
        print("Mean Recall:", round(cv_results['test_recall_weighted'].mean(), 4))
        print("Standard Deviation (Recall):", round(cv_results['test_recall_weighted'].std(), 4))
        print("Mean F1-Score:", round(cv_results['test_f1_weighted'].mean(), 4))
        print("Standard Deviation (F1-Score):", round(cv_results['test_f1_weighted'].std(), 4))
# determine_bagged_model(X_scaled, y, classifiers, 42)

# grid searching my best bagged models
def grid_search_best_params(model_grid, X, y, cv=5, n_iter=18, random_state=None):
    """
    Performs RandomizedSearchCV to find the best parameters for multiple models.

    Parameters:
    model_grid (dict): Dictionary where keys are model names,
                       values are tuples (model, parameter grid).
    X, y: Training data.
    cv (int): Number of cross-validation folds (default: 5).
    n_iter (int): Number of random search iterations (default: 20).
    random_state (int): Random seed (default: None).

    Returns:
    best_params_dict (dict): Dictionary containing the best parameters for each model.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

    best_params_dict = {}

    for name, (model, param_grid) in model_grid.items():
        print(f"Running RandomizedSearchCV for {name}...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring="f1_weighted",
            random_state=random_state,
            n_jobs=-1  # Parallel processing
        )
        print(f"Randomized Search Completed")

        random_search.fit(X_train, y_train)

        best_params = random_search.best_params_
        best_params_dict[name] = best_params

        print(f"Best parameters for {name}: {best_params}\n")
    return best_params_dict
#Model Dictionary
models = {
    "BaggedSVC": (
        BaggingClassifier(estimator=SVC(), n_jobs=-1),
        {
            "estimator__C": [0.1, 1, 10],
            "estimator__kernel": ["linear", "rbf"],
            "n_estimators": [10, 50, 100],
        }
    ),
    "BaggedRandomForest": (
        BaggingClassifier(estimator=RandomForestClassifier(), n_jobs=-1),
        {
            "estimator__n_estimators": [10, 50, 100],
            "estimator__max_depth": [None, 10, 20],
            "n_estimators": [10, 50, 100],
        }
    ),
    "BaggedDecisionTree": (
        BaggingClassifier(estimator=DecisionTreeClassifier(), n_jobs=-1),
        {
            "estimator__max_depth": [None, 5, 10, 20],
            "estimator__min_samples_split": [2, 5, 10],
            "n_estimators": [10, 50, 100],
        }
    )
}
# best_params = grid_search_best_params(models, X_scaled, y)

# Basic Model Evaluation
def evaluate_model(X, y, model, random_state=None):
    print(f"""
######################################
########## Model Evaluation ##########
######################################""")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_state)

    for name, clf in model.items():
        kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)

        # Store scores
        kfold_acc_scores = []
        kfold_precision_scores = []
        kfold_recall_scores = []
        kfold_f1_scores = []

        # Kfold testing each model
        for train_index, val_index in kfold.split(X_train):
            X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            # Train on Training Fold
            clf.fit(X_train_fold, y_train_fold)

            # Validate on Validation Fold
            y_pred = clf.predict(X_val_fold)

            # Store metrics with weighted average
            kfold_acc_scores.append(accuracy_score(y_val_fold, y_pred))
            kfold_precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))
            kfold_recall_scores.append(recall_score(y_val_fold, y_pred, average='weighted'))
            kfold_f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))

        # Train on whole data,
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val)
        y_test_pred = clf.predict(X_test)

        ########## Printing Results ##########
        # Metrics Calculation
        metrics = {
            "Accuracy": [
                np.mean(kfold_acc_scores), np.std(kfold_acc_scores),
                accuracy_score(y_val, y_val_pred),
                accuracy_score(y_test, y_test_pred)
            ],
            "Precision": [
                np.mean(kfold_precision_scores), np.std(kfold_precision_scores),
                precision_score(y_val, y_val_pred, average='weighted'),
                precision_score(y_test, y_test_pred, average='weighted')
            ],
            "Recall": [
                np.mean(kfold_recall_scores), np.std(kfold_recall_scores),
                recall_score(y_val, y_val_pred, average='weighted'),
                recall_score(y_test, y_test_pred, average='weighted')
            ],
            "F1-Score": [
                np.mean(kfold_f1_scores), np.std(kfold_f1_scores),
                f1_score(y_val, y_val_pred, average='weighted'),
                f1_score(y_test, y_test_pred, average='weighted')
            ]
        }

        # Print Table Header
        print(f"\n{name} Performance Evaluation:")
        print("-" * 85)
        print(f"{'Metric':<12} | {'Cross-Validation (Mean ± Std)':<33} | {'Validation Set':<15} | {'Test Set':<15}")
        print("-" * 85)

        # Print Metrics
        for metric, values in metrics.items():
            print(f"{metric:<12} | {values[0]:<18.4f} ± {values[1]:<6.4f} | {values[2]:<15.4f} | {values[3]:<15.4f}")

        print("-" * 85)
LGBM_params = {
        'objective': 'multiclass',  # Or 'multiclass', 'regression', etc.
        'metric': 'multi_logloss',  # Or appropriate metric
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_estimators': 100,
        'min_child_samples': 20,
        'min_child_weight': 1e-3,
        'min_split_gain': 0,
        'random_state': None
    }
ensemble_classifiers = [LogisticRegression(solver='liblinear'),LGBMClassifier(**LGBM_params),
                        SVC(kernel='linear', C=10), RandomForestClassifier(n_estimators=100, max_depth=20),
                        DecisionTreeClassifier(max_depth=20, min_samples_split=2), KNeighborsClassifier(n_neighbors=5)]
chosen_classifiers = {
    # "Bagged SVC Classifier": BaggingClassifier(estimator=SVC(kernel='linear', C=10), n_estimators=100),
    # "Bagged Random Forest Classifier": BaggingClassifier(estimator=RandomForestClassifier(n_estimators=100, max_depth=20), n_estimators=100),
    "Bagged Decision Tree Classifier": BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=20, min_samples_split=2), n_estimators=50),
    "KNN Neighbors Classifier": KNeighborsClassifier(n_neighbors=5),
    "Light GBM Classifier": LGBMClassifier(**LGBM_params),
    "Ensemble Vote Classifier": EnsembleVoteClassifier(clfs=ensemble_classifiers, voting='hard'),
    "Logistic Regressor": LogisticRegression(solver='liblinear', max_iter=1000)
}
evaluate_model(best_X, y, chosen_classifiers, None)

#########################################################
########## SINGLE MODEL EVALUATION FOR TESTING ##########
#########################################################

# Basic logistics
def logistics_regression(X, y, random_state=None):
    print(f"""
####################################################
########## Logistic Regression Evaluation ##########
####################################################""")
    # Step 1: Split into Train (60%), Validation (20%), and Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

    clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=random_state)

    # Step 2: Perform K-Fold Cross-Validation on Training Data (60%)
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Store scores
    kfold_acc_scores = []
    kfold_precision_scores = []
    kfold_recall_scores = []
    kfold_f1_scores = []

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train on Training Fold
        clf.fit(X_train_fold, y_train_fold)

        # Validate on Validation Fold
        y_pred = clf.predict(X_val_fold)

        # Store metrics with weighted average
        kfold_acc_scores.append(accuracy_score(y_val_fold, y_pred))
        kfold_precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))
        kfold_recall_scores.append(recall_score(y_val_fold, y_pred, average='weighted'))
        kfold_f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))

    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    y_test_pred = clf.predict(X_test)

    ########## Printing Results ##########
    # Metrics Calculation
    metrics = {
        "Accuracy": [
            np.mean(kfold_acc_scores),
            accuracy_score(y_val, y_val_pred),
            accuracy_score(y_test, y_test_pred)
        ],
        "Precision": [
            np.mean(kfold_precision_scores),
            precision_score(y_val, y_val_pred, average='weighted'),
            precision_score(y_test, y_test_pred, average='weighted')
        ],
        "Recall": [
            np.mean(kfold_recall_scores),
            recall_score(y_val, y_val_pred, average='weighted'),
            recall_score(y_test, y_test_pred, average='weighted')
        ],
        "F1-Score": [
            np.mean(kfold_f1_scores),
            f1_score(y_val, y_val_pred, average='weighted'),
            f1_score(y_test, y_test_pred, average='weighted')
        ]
    }

    # Print Table Header
    print(f"\nLogistic Regression Performance Evaluation:")
    print("-" * 65)
    print(f"{'Metric':<12} | {'Cross-Validation':<18} | {'Validation Set':<15} | {'Test Set':<15}")
    print("-" * 65)

    # Print Metrics
    for metric, values in metrics.items():
        print(f"{metric:<12} | {values[0]:<18.4f} | {values[1]:<15.4f} | {values[2]:<15.4f}")

    print("-" * 65)
# logistics_regression(best_X, y, random_state=None)

# KNN classifier
def KNN_neighbors(X, y, random_state=None):
    print(f"""
####################################################
########## KNN Neighbors Evaluation ##########
####################################################""")
    # Step 1: Split into Train (60%), Validation (20%), and Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

    clf = KNeighborsClassifier(n_neighbors=5)  # You can experiment with n_neighbors

    # Step 2: Perform K-Fold Cross-Validation on Training Data (60%)
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Store scores
    kfold_acc_scores = []
    kfold_precision_scores = []
    kfold_recall_scores = []
    kfold_f1_scores = []

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train on Training Fold
        clf.fit(X_train_fold, y_train_fold)

        # Validate on Validation Fold
        y_pred = clf.predict(X_val_fold)

        # Store metrics with weighted average
        kfold_acc_scores.append(accuracy_score(y_val_fold, y_pred))
        kfold_precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))
        kfold_recall_scores.append(recall_score(y_val_fold, y_pred, average='weighted'))
        kfold_f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))

    # Step 3: Display Cross-Validation Results
    print("\nCross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(kfold_acc_scores):.4f}")
    print(f"Mean Precision: {np.mean(kfold_precision_scores):.4f}")
    print(f"Mean Recall: {np.mean(kfold_recall_scores):.4f}")
    print(f"Mean F1-Score: {np.mean(kfold_f1_scores):.4f}")

    # Step 4: Train without Folding
    clf.fit(X_train, y_train)

    # Step 5: Predict on Validation (20%)
    y_val_pred = clf.predict(X_val)
    print("\nFinal Validation Set Evaluation:")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Validation Precision: {precision_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Validation Recall: {recall_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Validation F1-Score: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")

    # Step 6: Predict on Test (20%)
    y_test_pred = clf.predict(X_test)
    print("\nFinal Model Evaluation on Unseen Test Data:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"Test Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"Test F1-Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
# KNN_neighbors(best_X, y, random_state=None)

# LGBM Classifier
def LGBM_evaluation(X, y, random_state=None, **kwargs):
    """
    Evaluates an optimized LightGBM classifier with train/validation/test split and K-Fold cross-validation.

    Args:
        X: Features (pandas DataFrame or numpy array).
        y: Target variable (pandas Series or numpy array).
        random_state: Seed for reproducibility.
        **kwargs: Additional LightGBM parameters to override defaults.
    """
    print(f"""
####################################################
########## LightGBM Evaluation #####################
####################################################""")

    # Step 1: Split into Train (60%), Validation (20%), and Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

    params = {
        'objective': 'multiclass',  # Or 'multiclass', 'regression', etc.
        'metric': 'multi_logloss',  # Or appropriate metric
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_estimators': 100,
        'min_child_samples': 20,
        'min_child_weight': 1e-3,
        'min_split_gain': 0,
        'random_state': random_state
    }

    # Override default parameters with any provided kwargs
    params.update(kwargs)

    clf = LGBMClassifier(**params)

    # Step 2: Perform K-Fold Cross-Validation on Training Data (60%)
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Store scores
    kfold_acc_scores = []
    kfold_precision_scores = []
    kfold_recall_scores = []
    kfold_f1_scores = []

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train on Training Fold
        clf.fit(X_train_fold, y_train_fold)

        # Validate on Validation Fold
        y_pred = clf.predict(X_val_fold)

        # Store metrics with weighted average
        kfold_acc_scores.append(accuracy_score(y_val_fold, y_pred))
        kfold_precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))
        kfold_recall_scores.append(recall_score(y_val_fold, y_pred, average='weighted'))
        kfold_f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))

    # Step 3: Display Cross-Validation Results
    print("\nCross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(kfold_acc_scores):.4f}")
    print(f"Mean Precision: {np.mean(kfold_precision_scores):.4f}")
    print(f"Mean Recall: {np.mean(kfold_recall_scores):.4f}")
    print(f"Mean F1-Score: {np.mean(kfold_f1_scores):.4f}")

    # Step 4: Train without Folding on full train set.
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Step 5: Predict on Validation (20%)
    y_val_pred = clf.predict(X_val)
    print("\nFinal Validation Set Evaluation:")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Validation Precision: {precision_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Validation Recall: {recall_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Validation F1-Score: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")

    # Step 6: Predict on Test (20%)
    y_test_pred = clf.predict(X_test)
    print("\nFinal Model Evaluation on Unseen Test Data:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"Test Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"Test F1-Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
# LGBM_evaluation(best_X, y, random_state=None)

# Voting Ensemble
def voting_ensemble(X, y, random_state=None):
    print(f"""
####################################################
########## Voting Ensemble Evaluation ##############
####################################################""")
    # Step 1: Split into Train (60%), Validation (20%), and Test (20%)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=random_state)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=random_state)

    params = {
        'objective': 'multiclass',  # Or 'multiclass', 'regression', etc.
        'metric': 'multi_logloss',  # Or appropriate metric
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 20,
        'verbose': -1,
        'n_estimators': 100,
        'min_child_samples': 20,
        'min_child_weight': 1e-3,
        'min_split_gain': 0,
        'random_state': random_state
    }

    # Ensembles
    clf1 = LogisticRegression(solver='liblinear', random_state=random_state)
    clf2 = LGBMClassifier(**params)
    clf3 = SVC(kernel='linear', probability=True, random_state=random_state)
    clf4 = RandomForestClassifier(n_estimators=500, random_state=random_state)
    clf5 = DecisionTreeClassifier()
    clf6 = KNeighborsClassifier(n_neighbors=5)

    clf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4, clf5, clf6], voting='hard')

    # Step 2: Perform K-Fold Cross-Validation on Training Data (60%)
    kfold = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # Store scores
    kfold_acc_scores = []
    kfold_precision_scores = []
    kfold_recall_scores = []
    kfold_f1_scores = []

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train on Training Fold
        clf.fit(X_train_fold, y_train_fold)

        # Validate on Validation Fold
        y_pred = clf.predict(X_val_fold)

        # Store metrics with weighted average
        kfold_acc_scores.append(accuracy_score(y_val_fold, y_pred))
        kfold_precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))
        kfold_recall_scores.append(recall_score(y_val_fold, y_pred, average='weighted'))
        kfold_f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))

    # Step 3: Display Cross-Validation Results
    print("\nCross-Validation Results:")
    print(f"Mean Accuracy: {np.mean(kfold_acc_scores):.4f}")
    print(f"Mean Precision: {np.mean(kfold_precision_scores):.4f}")
    print(f"Mean Recall: {np.mean(kfold_recall_scores):.4f}")
    print(f"Mean F1-Score: {np.mean(kfold_f1_scores):.4f}")

    # Step 4: Train without Folding
    clf.fit(X_train, y_train)

    # Step 5: Predict on Validation (20%)
    y_val_pred = clf.predict(X_val)
    print("\nFinal Validation Set Evaluation:")
    print(f"Validation Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
    print(f"Validation Precision: {precision_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Validation Recall: {recall_score(y_val, y_val_pred, average='weighted'):.4f}")
    print(f"Validation F1-Score: {f1_score(y_val, y_val_pred, average='weighted'):.4f}")

    # Step 6: Predict on Test (20%)
    y_test_pred = clf.predict(X_test)
    print("\nFinal Model Evaluation on Unseen Test Data:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"Test Precision: {precision_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"Test Recall: {recall_score(y_test, y_test_pred, average='weighted'):.4f}")
    print(f"Test F1-Score: {f1_score(y_test, y_test_pred, average='weighted'):.4f}")
# voting_ensemble(best_X, y, random_state=None)

# Stacked Model
def stacked_model(X, y):
    def getUnfitModels():
        models = list()
        models.append(LogisticRegression())
        models.append(DecisionTreeClassifier())
        models.append(RandomForestClassifier())
        models.append(SVC(kernel='linear'))
        return models

    def evaluateModel(y_test, predictions, model):
        print("\n*** " + model.__class__.__name__)
        report = classification_report(y_test, predictions)
        print(report)

    def fitBaseModels(X_train, y_train, X_test, models):
        dfPredictions = pd.DataFrame()

        # Fit base model and store its predictions in dataframe.
        for i in range(0, len(models)):
            models[i].fit(X_train, y_train)
            predictions = models[i].predict(X_test)
            colName = str(i)
            dfPredictions[colName] = predictions
        return dfPredictions, models

    def fitStackedModel(X, y):
        params = {
            'objective': 'multiclass',  # Or 'multiclass', 'regression', etc.
            'metric': 'multi_logloss',  # Or appropriate metric
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 20,
            'verbose': -1,
            'n_estimators': 100,
            'min_child_samples': 20,
            'min_child_weight': 1e-3,
            'min_split_gain': 0
        }

        model = LGBMClassifier(**params)
        model.fit(X, y)
        return model

#     print(f"""
# ####################################################
# ########## Stacked Model Evaluation ################
# ####################################################""")

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Get base models.
    unfitModels = getUnfitModels()

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_val, unfitModels)
    stackedModel = fitStackedModel(dfPredictions, y_val)

    # Evaluate base models with validation data.
    # print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_test)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        # evaluateModel(y_test, predictions, models[i])

    # # Evaluate stacked model with validation data.
    # stackedPredictions = stackedModel.predict(dfValidationPredictions)
    # print("\n** Evaluate Stacked Model **")
    # evaluateModel(y_test, stackedPredictions, stackedModel)

    kfold = KFold(n_splits=5, shuffle=True, random_state=None)

    # Store scores
    kfold_acc_scores = []
    kfold_precision_scores = []
    kfold_recall_scores = []
    kfold_f1_scores = []

    for train_index, val_index in kfold.split(X_train):
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Train on Training Fold
        stackedModel.fit(X_train_fold, y_train_fold)

        # Validate on Validation Fold
        y_pred = stackedModel.predict(X_val_fold)

        # Store metrics with weighted average
        kfold_acc_scores.append(accuracy_score(y_val_fold, y_pred))
        kfold_precision_scores.append(precision_score(y_val_fold, y_pred, average='weighted'))
        kfold_recall_scores.append(recall_score(y_val_fold, y_pred, average='weighted'))
        kfold_f1_scores.append(f1_score(y_val_fold, y_pred, average='weighted'))

    y_val_pred = stackedModel.predict(X_val)
    y_test_pred = stackedModel.predict(X_test)

    ########## Printing Results ##########
    # Metrics Calculation
    metrics = {
        "Accuracy": [
            np.mean(kfold_acc_scores), np.std(kfold_acc_scores),
            accuracy_score(y_val, y_val_pred),
            accuracy_score(y_test, y_test_pred)
        ],
        "Precision": [
            np.mean(kfold_precision_scores), np.std(kfold_precision_scores),
            precision_score(y_val, y_val_pred, average='weighted'),
            precision_score(y_test, y_test_pred, average='weighted')
        ],
        "Recall": [
            np.mean(kfold_recall_scores), np.std(kfold_recall_scores),
            recall_score(y_val, y_val_pred, average='weighted'),
            recall_score(y_test, y_test_pred, average='weighted')
        ],
        "F1-Score": [
            np.mean(kfold_f1_scores), np.std(kfold_f1_scores),
            f1_score(y_val, y_val_pred, average='weighted'),
            f1_score(y_test, y_test_pred, average='weighted')
        ]
    }

    # Print Table Header
    print(f"\nStacked Model Performance Evaluation:")
    print("-" * 85)
    print(f"{'Metric':<12} | {'Cross-Validation (Mean ± Std)':<33} | {'Validation Set':<15} | {'Test Set':<15}")
    print("-" * 85)

    # Print Metrics
    for metric, values in metrics.items():
        print(f"{metric:<12} | {values[0]:<18.4f} ± {values[1]:<6.4f} | {values[2]:<15.4f} | {values[3]:<15.4f}")

    print("-" * 85)
stacked_model(best_X, y)
