# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the aesthetic style of the plots
sns.set(style="whitegrid")

# Task 1: Load and Explore the Dataset
try:
    # Load the Iris dataset
    from sklearn.datasets import load_iris
    iris_data = load_iris()
    
    # Create a DataFrame
    iris_df = pd.DataFrame(data=iris_data.data, columns=iris_data.feature_names)
    iris_df['species'] = iris_data.target

    # Map target integers to species names
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

except Exception as e:
    print(f"Error loading the dataset: {e}")

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(iris_df.head())

# Explore the structure of the dataset
print("\nDataset Info:")
print(iris_df.info())

print("\nMissing values in each column:")
print(iris_df.isnull().sum())

# Clean the dataset (if necessary)
# In this case, the Iris dataset has no missing values.
# If there were missing values, you could fill or drop them here.
# Example: iris_df.dropna(inplace=True)

# Task 2: Basic Data Analysis
# Compute basic statistics of the numerical columns
print("\nDescriptive Statistics:")
print(iris_df.describe())

# Perform groupings on a categorical column
mean_petal_length = iris_df.groupby('species')['petal length (cm)'].mean()
print("\nMean Petal Length per Species:")
print(mean_petal_length)

# Identify patterns or interesting findings
observations = """
1. The average petal length differs significantly across species.
2. Setosa has the smallest petal length, while Virginica has the largest.
"""
print("\nFindings and Observations:")
print(observations)

# Task 3: Data Visualization
# Line Chart (not applicable for Iris dataset, so we will skip it)
# For demonstration, let's assume we have some time-series data.

# Bar Chart: Average Petal Length per Species
plt.figure(figsize=(8, 5))
mean_petal_length.plot(kind='bar', color='skyblue')
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.xticks(rotation=0)
plt.grid(axis='y')
plt.show()

# Histogram: Distribution of Petal Length
plt.figure(figsize=(10, 6))
plt.hist(iris_df['petal length (cm)'], bins=15, color='lightgreen', edgecolor='black')
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.grid(axis='y')
plt.show()

# Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(10, 6))
sns.scatterplot(data=iris_df, x='sepal length (cm)', y='petal length (cm)', hue='species', style='species', s=100)
plt.title('Sepal Length vs. Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.grid()
plt.show()

# Final Observations
final_observations = """
1. The bar chart shows that Virginica has the largest average petal length.
2. The histogram indicates that most petal lengths are concentrated around 1.5 to 5.0 cm.
3. The scatter plot shows a clear separation between species based on petal length and sepal length, indicating that these features are useful for classification tasks.
"""
print("\nFinal Observations:")
print(final_observations)