import pandas as pd
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
sub_order = ['no', 'Sometimes', 'Frequently', 'Always']

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
obesity_df = obesity_df[[""]]