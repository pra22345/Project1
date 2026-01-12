import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr

wr.filterwarnings('ignore')
sns.set_style("darkgrid")

# Load dataset
df = pd.read_csv('WineQT.csv')

# Basic inspection
print(df.head())
print(df.shape)
print(df.info())
print(df.describe().T)

# Quality distribution
quality_counts = df['quality'].value_counts()
plt.figure(figsize=(8, 6))
plt.bar(quality_counts.index, quality_counts, color='deeppink')
plt.title('Count Plot of Quality')
plt.xlabel('Quality')
plt.ylabel('Count')
plt.show()

# Histograms with skewness
numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
plt.figure(figsize=(14, len(numerical_columns) * 3))
for idx, feature in enumerate(numerical_columns, 1):
    plt.subplot(len(numerical_columns), 2, idx)
    sns.histplot(df[feature], kde=True)
    plt.title(f"{feature} | Skewness: {round(df[feature].skew(), 2)}")
plt.tight_layout()
plt.show()

# Swarm plot
plt.figure(figsize=(10, 8))
sns.swarmplot(x="quality", y="alcohol", data=df, palette='viridis')
plt.title('Swarm Plot for Quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()

# Pair plot
sns.pairplot(df, height=2.5)
plt.suptitle('Pair Plot for DataFrame', y=1.02)
plt.show()

# Violin plot
df['quality'] = df['quality'].astype(str)
plt.figure(figsize=(10, 8))
sns.violinplot(x="quality", y="alcohol", data=df,
               palette={'3': 'lightcoral', '4': 'lightblue', '5': 'lightgreen',
                        '6': 'gold', '7': 'lightskyblue', '8': 'lightpink'},
               alpha=0.7)
plt.title('Violin Plot for Quality and Alcohol')
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.show()

# Boxplot
plt.figure(figsize=(10, 8))
sns.boxplot(x='quality', y='alcohol', data=df)
plt.title('Box Plot for Quality and Alcohol')
plt.show()

# Correlation heatmap
plt.figure(figsize=(15, 10))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='Pastel2', linewidths=2)
plt.title('Correlation Heatmap')
plt.show()