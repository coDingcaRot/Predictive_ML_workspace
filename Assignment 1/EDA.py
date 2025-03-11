import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder

# Initial Loading values and setup
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

def split_categorical_numerical(df):
    """
    Splits up the dataframe between categorical and numerical columns
    :param df:
    :return:
    """
    categorical_df = df.select_dtypes(include=['object'])
    numerical_df = df.select_dtypes(exclude=['object'])
    return categorical_df, numerical_df

def initial_info(df, title):
    """
    Displays the initial information of a given dataset
    :param df: pandas dataframe
    :param title: title of the dataset
    """
    # Initial view
    print(f"""
{title} Dataset first 5 values
    """)
    print(df.head(5))
    print(f"""
{title} Dataset Statistics
    """)
    print(df.describe())
    print(f"""
{title} Dataset Information
    """)
    print(df.info())
    print("Shape of Dataset", df.shape)
    print("Total Duplicates in Dataset", df.duplicated().sum())
    print(f"{title} Dataset Column Missing Values \n{df.isna().sum()}")
# initial_info(obesity_df, "Obesity Dataset")

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
cat_df, num_df = split_categorical_numerical(obesity_df)

############################################
########## UNI VARIATE ANALYSIS ############
############################################

def showcase_histogram(df, column_count, row_count):
    """
    Displays subplots of numerical dataframe. Iterates through the numerical pandas dataframe and plots the histogram for the values
    :param df: numerical pandas dataframe
    :param column_count: Column count in the dataframe
    :param row_count: Row count that's divisiable by the column count
    """
    subplot_row = row_count
    subplot_col = column_count # Simplified calculation

    plt.figure(figsize=(subplot_col * 6, subplot_row * 6))  # Adjust figure size
    counter = 1
    for x in df.columns:
        plt.subplot(subplot_row, subplot_col, counter)
        plt.hist(df[x])
        plt.title(f"{x} Histogram", fontsize=20)
        plt.xlabel(f"{x}")
        plt.ylabel(f"count")
        counter += 1
    plt.tight_layout(pad=2.0)
    plt.show()
# showcase_histogram(num_df, 3, 3)

def showcase_barplots(df, column_count, row_count):
    """
    Displays subplots of categorical dataframe. Iterates through the categorical pandas dataframe and plots the bar plots for the values
    :param df: categorical pandas dataframe
    :param column_count: Column count in the dataframe
    :param row_count: Row count that's divisible by the column count
    """
    subplot_row = row_count
    subplot_col = math.ceil(column_count / row_count)

    plt.figure(figsize=(subplot_col * 6, subplot_row * 6))
    counter = 1
    for x in df.columns:
        plt.subplot(subplot_row, subplot_col, counter)
        counts = df[x].value_counts()  # Count occurrences of each category
        plt.bar(counts.index, counts.values)  # Create bar plot

        plt.title(f"Distribution of {x}") #More descriptive title.
        plt.xlabel(f"{x} Categories") # x axis is categories.
        plt.ylabel("Count") #y axis is count.
        plt.xticks(rotation=45, ha='right') #rotate x axis labels if needed.

        counter += 1

    plt.tight_layout(pad=2.0)
    plt.show()
# showcase_barplots(cat_df, cat_df.shape[1], 3)

############################################
##########  MULTIVARIATE ANALYSIS ##########
############################################

def ordinal_encoding(dataframe):
    dataframe = dataframe.copy()

    # Encode categorical variables to be able to view in heatmap
    label_encoder = LabelEncoder()
    for col in dataframe.columns:
        if dataframe[col].dtypes == 'object':
            label_encoder.fit(dataframe[col])
            dataframe[col] = label_encoder.transform(dataframe[col])

    return dataframe

# OVERALL
def corr_heatmap(df):
    """
    correlation heatmap visualization
    It checks the dataframe for any categorical variables
    and uses ordinal encoding in order to visually
    see which best correlates with our target
    :param df: as a pandas dataframe
    :return: None
    """
    dataframe = ordinal_encoding(df)

    # Plot correlation heatmap
    plt.figure(figsize=(14, 10))
    plt.title("Numerical Heatmap")
    sns.heatmap(dataframe.corr(), annot=True, cmap="Blues")
    plt.show()
# corr_heatmap(num_df)

# irrelvent sinces we are using a wrong method of comparison
def showcase_scatter_matrix(df):
    """
    Displays a scatter matrix using seaborns pairplot.

    :param df: pandas dataframe
    """
    dataframe = df.copy()
    sns.pairplot(dataframe.sample(500))
    plt.show()
# num_df = pd.concat([num_df, obesity_df["obesity_status"]], axis=1)
# num_df = ordinal_encoding(num_df)
# showcase_scatter_matrix(num_df)

# NUMERICAL MULTIVARIATES
def boxplot_comparison(df, row_count, column_count, target="obesity_status"):
    subplot_row = row_count
    subplot_col = column_count  # Ensure proper subplot arrangement

    plt.figure(figsize=(subplot_col * 6, subplot_row * 6))
    counter = 1

    for col in df.columns:
        if col == target:  # Ensure we skip the target column
            continue

        plt.subplot(subplot_row, subplot_col, counter)
        plt.title(f"{target} vs {col} Box Plot")

        sns.boxplot(data=df, x=target, y=col, palette="GnBu")
        plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
        counter += 1
    plt.tight_layout()
    plt.show()
# boxplot_comparison(obesity_df[["Age", "Weight", "daily_water_drinking", "obesity_status", "food_between_meals"]], 2, 2)

def violin_plot_comparison(df, row_count, column_count, target_order, target="obesity_status"):
    subplot_row = row_count
    subplot_col = column_count  # Ensure proper subplot arrangement

    plt.figure(figsize=(subplot_col * 6, subplot_row * 6))
    counter = 1


    for col in df.columns:
        if col == target:  # Ensure we skip the target column
            continue

        plt.subplot(subplot_row, subplot_col, counter)
        plt.title(f"{target} vs {col} Violin Plot", fontsize=15)

        sns.violinplot(data=df, x=target, y=col, palette="GnBu", order=target_order)
        plt.xticks(rotation=45, ha="right")  # Rotate labels for readability
        counter += 1
    plt.tight_layout()
    plt.show()
# violin_plot_comparison(obesity_df[["Age", "Weight", "daily_water_drinking", "obesity_status"]], 2, 2,
#                        target_order=order)

# CATEGORICAL MULTIVARIATES
def count_plot_comparison(df, row_count, column_count, target_order, target="obesity_status"):
    """
    Creates a subplot of comparisons between the target and categorical variables
    :param target_order: order param to order our categorical ordinal values.
    :param df: pandas dataframe
    :param row_count: Rows for subplot
    :param column_count: Columns for subplot
    :param target: our target variable
    :return: none
    """
    subplot_row = row_count
    subplot_col = column_count

    plt.figure(figsize=(subplot_col * 6, subplot_row * 6))
    counter = 1
    for col in df.columns:
        if col == f"{target}":
            continue

        plt.subplot(subplot_row, subplot_col, counter)
        plt.title(f"{target} vs {col} count plot")

        sns.countplot(data=df, x=target,
                      hue=col, palette=sns.color_palette("GnBu", n_colors=len(df[col].unique())), order=target_order)
        plt.xticks(rotation=45, ha="right")
        counter += 1
    plt.tight_layout()
    plt.show()
# count_plot_comparison(obesity_df[["food_between_meals", "family_history_with_overweight",
#                                   "freq_high_calorie_food_eat", "transportation_method",
#                                   "Gender", "SMOKE",
#                                   "caloric_monitoring", "freq_alcohol_usage",
#                                   "obesity_status"]],
#                       3, 3, target_order=order)

# singular stacked_bar
def plot_stacked_bar(df, target_col, other_col):
    """
    Plots a stacked bar chart to show the proportions of the target variable.

    Args:
        df: Pandas DataFrame containing the data.
        target_col: Name of the target categorical column.
        other_col: Name of the other categorical column.
    """

    sns.histplot(data=df, x=other_col, hue=target_col, multiple='fill')
    plt.title(f"Proportions of {target_col} by {other_col}")
    plt.xlabel(other_col)
    plt.ylabel("Proportion")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()
# plot_stacked_bar(obesity_df, "obesity_status", "family_history_with_overweight")

sub_order = ['no', 'Sometimes', 'Frequently', 'Always']

# sns.scatterplot(data=obesity_df, x="Weight", y="Height", hue="obesity_status")
# sns.countplot(data=obesity_df, x="food_between_meals", hue="obesity_status",
#               palette=sns.color_palette("GnBu", n_colors=len(obesity_df["obesity_status"].unique())),
#               order=sub_order)
# sns.pairplot(data=obesity_df, hue="obesity_status")
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.show()

# Average age for
 # obesity_df.groupby("obesity_status")["Age"].median().sort_values(ascending=True).plot(kind="bar",color = sns.color_palette("GnBu")  )
sns.clustermap(data=num_df)
# sns.stripplot(data=obesity_df, x="Gender", y="Age",
#               hue="obesity_status", palette="crest")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()



#########################################
##########  DATA PREPORCESSING ##########
#########################################

def onehot_encode(df):
    dataframe = df.copy()  # Create a copy of the original DataFrame to avoid modifying the original data

    # Loop through columns and one-hot encode only categorical variables
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':  # Check if the column is categorical
            # One-hot encode and drop the original column
            encoded_cols = pd.get_dummies(dataframe[col], prefix=col)
            dataframe = dataframe.drop(col, axis=1)  # Drop the original categorical column
            dataframe = pd.concat([dataframe, encoded_cols], axis=1)  # Add the encoded columns

    return dataframe
def ordinal_encode(df):
    dataframe = df.copy()

    # iterate through each column and check if it can be encoded
    # if it can be get its unique values, and create a mapping
    # used the map values and replace the column with the new ordinal values
    for col in dataframe.columns:
        if dataframe[col].dtype == 'object':
           order = dataframe[col].unique()
           # Create a mapping of categories to numbers based on the order
           mapping = {category: idx for idx, category in enumerate(order)}
           # Apply the mapping to the column
           dataframe[col] = dataframe[col].map(mapping)

    return dataframe
def smart_encode_categorical(df, ordinal_cols=None):
    """
    Encodes categorical columns in a DataFrame based on whether they are ordinal or nominal.

    Args:
        df: The pandas DataFrame.
        ordinal_cols: A list of column names that should be treated as ordinal.
                      If None, all 'object' type columns are treated as nominal.

    Returns:
        A new DataFrame with categorical columns encoded.
    """

    df_encoded = df.copy()

    # If ordinal_cols is not provided, treat all object columns as nominal
    if ordinal_cols is None:
        ordinal_cols = df

    # One-hot encode nominal columns
    nominal_cols = [col for col in df_encoded.select_dtypes(include=['object']).columns if col not in ordinal_cols]
    for col in nominal_cols:
        df_encoded = pd.get_dummies(df_encoded, columns=[col], prefix=[col], drop_first=True, dtype="float64")

    # Label encode ordinal columns
    label_encoder = LabelEncoder()
    for col in ordinal_cols:
        if df_encoded[col].dtype == 'object':  # Ensure it's still an object type
            df_encoded[col] = label_encoder.fit_transform(df_encoded[col])

    return df_encoded
