# Utilities
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Classifiers or Regressors + Utilities
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from mlxtend.classifier      import EnsembleVoteClassifier
from sklearn.ensemble        import AdaBoostClassifier, GradientBoostingClassifier
from xgboost                 import XGBClassifier, plot_importance

from sklearn.model_selection import train_test_split, KFold, cross_validate  # test train split
from sklearn import metrics #confusion matrix
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
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

###################################
########## Preprocessing ##########
###################################
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
    "SVC Classifier": SVC(kernel='linear'),
    "Random Forest Classifier": RandomForestClassifier(n_estimators=1000),
    "Decision Tree Classifier": DecisionTreeClassifier(), #takes a long time
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "XGB Classifier": XGBClassifier(),
}

def feature_selection(X, y, top_feature_num, random_state=None):
    print(f"""\n
############################################
########## Feature Selection List ##########
############################################
""")

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "Ridge Classifier": RidgeClassifier(),
        "SVC Classifier": SVC(kernel='linear'),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
        "Gradient Boosting Classifier": GradientBoostingClassifier(),
        "XGB Classifier": XGBClassifier(),
    }

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
    aggregated_df = pd.DataFrame({'Feature': X.columns, 'Mean Importance': mean_importance})
    aggregated_df = aggregated_df.sort_values(by='Mean Importance', ascending=False)
    print("\nMean Feature Importance Between All Classifiers:")
    print(aggregated_df.head(top_feature_num))

X = obesity_df_cleaned.copy()
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

y = X["obesity_status"]
del X["obesity_status"]
del X_scaled["obesity_status"]

# feature_selection(X_scaled, y, 5, None)

#####################################
########## MODEL SELECTION ##########
#####################################

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
            random_state=random_state
        )
        bagging_clf.fit(X_train, y_train)

        # Validation Set Evaluation
        y_val_pred = bagging_clf.predict(X_val)
        print("\nValidation Set Evaluation:")
        print("Precision:", precision_score(y_val, y_val_pred, average='weighted'))
        print("Recall:", recall_score(y_val, y_val_pred, average='weighted'))
        print("F1-Score:", f1_score(y_val, y_val_pred, average='weighted'))
        print(classification_report(y_val, y_val_pred))

        # Test Set Evaluation (Unseen Data)
        y_test_pred = bagging_clf.predict(X_test)
        print("\nTest Set Evaluation (Unseen Data):")
        print("Precision:", precision_score(y_test, y_test_pred, average='weighted'))
        print("Recall:", recall_score(y_test, y_test_pred, average='weighted'))
        print("F1-Score:", f1_score(y_test, y_test_pred, average='weighted'))
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
determine_bagged_model(X_scaled, y, classifiers, 42)

# # Grid searching
# def determine_voting_model():
#
# # Grid searching
# def determine_stacked_model():





