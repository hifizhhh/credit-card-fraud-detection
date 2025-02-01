# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, recall_score

# Set Seaborn style for better visualization
sns.set()
%matplotlib inline

# Load Dataset
file_path = 'C:/Bismillah uhuy/Credit Card Fraud Detection/Dataset Credit Card Fraud Detection/creditcard.csv'
df = pd.read_csv(file_path)

# Display Basic Information about the Dataset
print("Dataset Shape:", df.shape)
print("\nFirst 5 Rows of the Dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nDescriptive Statistics:")
print(df.describe())

# Class Distribution
class_names = {0: 'Not Fraud', 1: 'Fraud'}
print("\nClass Distribution:")
print(df['Class'].value_counts().rename(index=class_names))

# Visualize Features (V1-V28 and Amount)
plt.figure(figsize=(15, 12))
for i, column in enumerate(df.columns[1:29], 1):
    plt.subplot(5, 6, i)
    plt.plot(df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Prepare Features and Target
feature_names = df.columns[1:30]  # Features (V1-V28 and Amount)
target_name = df.columns[30]      # Target (Class)
X = df[feature_names]             # Feature data
y = df[target_name]               # Target data

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
print("\nTraining and Testing Set Sizes:")
print(f"Training set (X_train): {len(X_train)} samples")
print(f"Testing set (X_test): {len(X_test)} samples")

# Train Logistic Regression Model
model = LogisticRegression(random_state=1)
model.fit(X_train, y_train)

# Make Predictions
y_pred = model.predict(X_test)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Not Fraud', 'Fraud'], columns=['Not Fraud', 'Fraud'])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_df, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.show()

# Evaluate Model Performance
f1 = round(f1_score(y_test, y_pred), 2)
recall = round(recall_score(y_test, y_pred), 2)
print("\nModel Evaluation Metrics:")
print(f"F1 Score: {f1}")
print(f"Recall (Sensitivity): {recall}")