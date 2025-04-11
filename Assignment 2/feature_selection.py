import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Load dataset
train_df = pd.read_csv("./data/train.csv")

# Convert target to binary for correlation
train_df['HighRisk'] = train_df['HighRisk'].map({'Yes': 1, 'No': 0})

# Encode all object-type columns temporarily for correlation
encoded_df = train_df.copy()
for col in encoded_df.select_dtypes(include=['object']).columns:
    encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col].astype(str))

# Correlation with target
correlations = encoded_df.corr(numeric_only=True)['HighRisk'].drop("HighRisk").sort_values(key=abs, ascending=False)
print("Top correlated features with HighRisk:\n")
print(correlations.head(15))

# Optional: Plot heatmap of top features
top_features = correlations.head(10).index.tolist() + ['HighRisk']
sns.heatmap(encoded_df[top_features].corr(), annot=True, cmap='coolwarm')
plt.title("Top 10 Features Correlated with HighRisk")
plt.tight_layout()
plt.savefig("eda_top_features_heatmap.png")
plt.show()

# Save to file for report
correlations.to_csv("./data/eda_feature_importance.csv")
