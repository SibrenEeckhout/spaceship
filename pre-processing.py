import pandas as pd
import matplotlib.pyplot as plt

# read the CSV file
df = pd.read_csv('train.csv')



# Define a function to label encode categorical columns
def label_encode(df, col):
    mapping = {val: i for i, val in enumerate(df[col].unique())}
    df[col] = df[col].replace(mapping)
    return df

# Apply label encoding to categorical columns
categorical_cols = ["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP", "Name", "Transported"]
for col in categorical_cols:
    df = label_encode(df, col)

# Calculate the correlation matrix
corr_matrix = df.corr()
print(corr_matrix)
# Get the most highly correlated features
highly_correlated = corr_matrix.abs().unstack().sort_values(ascending=False).drop_duplicates()[:10]
highly_correlated = highly_correlated[highly_correlated > 0.5]
print(highly_correlated)
# Extract the index labels of the highly correlated features
highly_correlated_cols = [pair[0] for pair in highly_correlated.index.tolist()]
print(highly_correlated_cols)
# Create a correlation matrix of the highly correlated features
corr_matrix = df[highly_correlated_cols].corr()
print(corr_matrix)

# Create a heatmap of the correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
heatmap = ax.imshow(corr_matrix, cmap='coolwarm', vmin=0.5, vmax=1)
ax.set_xticks(range(len(corr_matrix.columns)))
ax.set_xticklabels(corr_matrix.columns, rotation=45)
ax.set_yticks(range(len(corr_matrix.columns)))
ax.set_yticklabels(corr_matrix.columns)
ax.set_title('Correlation Matrix Heatmap')
plt.colorbar(heatmap)
plt.show()


