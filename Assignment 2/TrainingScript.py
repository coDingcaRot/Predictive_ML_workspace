import pandas as pd
from sklearn.model_selection import test_train_split

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

heart_risk = pd.read_csv('HeartRisk.csv')


