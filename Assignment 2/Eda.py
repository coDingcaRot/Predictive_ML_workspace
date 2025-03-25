import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

heart_risk_df = pd.read_csv('HeartRisk.csv')

"""########## EDA PROCESS ##########"""

# INITIAL EDA
def split_categorical_numerical(df):
    """
    Splits up the dataframe between categorical and numerical columns
    :param df: pandas dataframe
    :return: categorical_df of object type dataframe, numerical_df of number type dataframe
    """
    categorical_df = df.select_dtypes(include=['object'])
    numerical_df = df.select_dtypes(exclude=['object'])
    print(f"Categorical Column Count: {categorical_df.shape[1]}")
    print(f"Numerical Column Count: {numerical_df.shape[1]}")
    return categorical_df, numerical_df
def initial_info(df, head_count, title):
    """
    Displays the initial information of a given dataset
    :param df: pandas dataframe
    :param title: title of the dataset
    """
    # Initial view
    print(f"""
{title} Dataset First {head_count} values
{df.head(head_count)}

{title} Dataset Statistics
{df.describe()}

{title} Dataset Information
{df.info()}

{title} Dataset Shape: {df.shape}

{title} Dataset Column Missing Values \n{df.isna().sum()}
""")
# initial_info(heart_risk_df, 5, "Heart Risk")
cat_df, num_df = split_categorical_numerical(heart_risk_df)

# UNIVARIATE EDA
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
# Revised one does not work too well
def display_histograms(df, display_rows, display_cols):
    """
    Takes in a numerical type dataframe and displays the histograms of each value.
    :param df: pandas dataframe
    :return: none
    """
    sns.set_palette("rocket")
    fig, ax = plt.subplots(display_rows, display_cols ,figsize=(display_rows * 4, display_cols * 4))
    ax = ax.flatten()
    for i, col in enumerate(df.columns):
        sns.countplot(df, x=col, ax=ax[i])
        ax[i].set_title(f"{col} Histogram")
    plt.tight_layout()
    plt.show()
# display_histograms(num_df, display_rows=2, display_cols=3)
showcase_histogram(num_df, 3, 2)

def showcase_barplots(df, column_count, row_count):
    """
    Displays subplots of categorical dataframe. Iterates through the categorical pandas dataframe and plots the bar plots for the values
    :param df: categorical pandas dataframe
    :param column_count: Column count in the dataframe
    :param row_count: Row count that's divisible by the column count
    """
    subplot_row = row_count
    subplot_col = column_count

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
showcase_barplots(cat_df, 9, 4)

# MULTIVARIATE EDA
