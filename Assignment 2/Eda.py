import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.model_selection import train_test_split
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Load the dataset
df = pd.read_csv("HeartRisk.csv")

# Check for structure and preview
# df.info(), df.head()

# Full numerica and category
df_num = df.select_dtypes(include=['number'])
df_cat = df.select_dtypes(exclude=['number'])
binary_cols = []
cat_cols = []
for col in df_cat.columns:
   unique_count = df_cat[col].nunique()
   if unique_count <= 2:
       binary_cols.append(col)
   else:
     cat_cols.append(col)
df_bin = df_cat[binary_cols]
df_cat = df_cat[cat_cols]

# # Samples
df_num_sample = df_num.sample(frac=0.1, random_state=42)
df_bin_sample = df_bin.sample(frac=0.1, random_state=42)
df_cat_sample = df_cat.sample(frac=0.1, random_state=42)


print(f'Numeric Sample {len(df_num_sample.columns)}')
print(f'Binary Sample {len(df_bin_sample.columns)}')
print(f'Categorical Sample {len(df_cat_sample.columns)}')

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

# showcase_barplots(df_bin_sample, 5, 5)
showcase_barplots(df_cat_sample, 4, 3)


# print(df_bin.head(10))
# df_num.info(), df_cat.info()
# print('Total Numeric columns', df_num.shape[1])
# print('Total Categorical Columns', df_cat.shape[1])


# binary_map = {'Yes': 1, 'No': 0}
# y_sample = df_cat_sample['HighRisk'].map(binary_map)
# # print(y_sample.unique())

# df_num_sample_y = pd.concat([df_num, y_sample], axis=1)
# # print(df_num_sample_y.head(5))
# print(df_cat_sample.head(5))
# corr = df_num_sample_y.corr()
# corr = corr.sort_values(by=['HighRisk'], ascending=False)
# # Figure size and heatmap visuals
# plt.figure(figsize=(10,10))
# sns.heatmap(corr[['HighRisk']], cmap='rocket', annot=True, )
# # plt.title(f"Numerical Data Fields Heatmap")
# plt.xticks(rotation=45)  # Rotate x-axis labels if needed
# plt.yticks(rotation=45)
# plt.show()

# train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["HighRisk"])

# # Save the datasets
# train_path = "./data/train.csv"
# test_path = "./data/test.csv"
# train_df.to_csv(train_path, index=False)
# test_df.to_csv(test_path, index=False)
#
# train_path, test_path