import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
df = pd.read_csv("hospital_readmissions.csv")
#df = df.sample(n=1000, random_state=42)  # use only 1000 rows for faster training
print("Initial Shape:", df.shape)
# Select limited features
df = df[[
    "age","time_in_hospital","n_lab_procedures","n_procedures","n_medications", "n_outpatient","n_inpatient",
    "n_emergency", "readmitted"
]]
  # keep this small for now
# Convert categorical features
#df = pd.get_dummies(df, drop_first=True)
#unique values 
print("\n🔍 Unique value counts in object columns:")
print(df.select_dtypes(include='object').nunique())

# Step 3: Data Preprocessing

# 3.1 Print and Handle missing values
print("\nMissing values per column:\n", df.isnull().sum())

# Correct way without chained assignment
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])   # fill with mode
    else:
        df[col] = df[col].fillna(df[col].median())    # fill with median

'''# 3.2 Remove duplicates
print("Shape after cleaning:", df.shape)'''
# Handling missing values correctly and safely
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# Remove duplicates
# df = df.drop_duplicates()
print("Shape after cleaning:", df.shape)
plt.figure(figsize=(12, 8))
# Only include numeric columns
numeric_df = df.select_dtypes(include=['int64', 'float64'])

# Plot box plots for all numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(16, 10))
for idx, col in enumerate(numeric_cols):
    plt.subplot(3, 3, idx+1)
    sns.boxplot(data=df, x=col, color='skyblue')
    plt.title(f'Boxplot of {col}')
    plt.tight_layout()

plt.suptitle(" Box Plots for Outlier Detection", fontsize=16, y=1.02)
plt.show()
# 3.3 Encode categorical columns except target
# First, make sure target_column is set to a column that actually exists in your dataframe
# For example, if your target column is named differently, update this line:
target_column = 'readmitted'  # Replace with the actual column name in your df
df['readmitted'] = df['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

# Numerical features
X = df.drop(target_column, axis=1)
y = df[target_column]
print(X.shape)
# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include='object').columns

# Scale numeric columns
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# Encode categorical columns using LabelEncoder
le = LabelEncoder()
for col in categorical_cols:
    X[col] = le.fit_transform(X[col])

# 3.6 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Exploratory Data Analysis (EDA)
plt.figure(figsize=(12, 8))
numeric_df = df.select_dtypes(include=['number'])  # <- FIX
sns.heatmap(numeric_df.corr(), cmap='coolwarm', annot=True)
plt.title("Correlation Heatmap")
plt.show()


# Visualizing target distribution
sns.countplot(x=target_column, data=df)  # Changed from y to use the actual column in df
plt.title(f"{target_column} Distribution")
plt.xlabel(f"{target_column} (Encoded)")
plt.ylabel("Count")
plt.show()

# Step 5: Model Selection — Random Forest (Classification)
model = RandomForestClassifier(random_state=42)

# Step 6: Model Training
model.fit(X_train, y_train)

# Step 7: Model Evaluation
y_pred = model.predict(X_test)
print("\n Evaluation Metrics:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Model Improvement — Hyperparameter Tuning with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_preds = best_model.predict(X_test)

print("\n Best Parameters from GridSearchCV:", grid_search.best_params_)
print("Improved Accuracy:", accuracy_score(y_test, best_preds))
# Plot box plots for all numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

plt.figure(figsize=(16, 10))
for idx, col in enumerate(numeric_cols):
    plt.subplot(3, 3, idx+1)
    sns.boxplot(data=df, x=col, color='skyblue')
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.suptitle(" Box Plots for Outlier Detection", fontsize=16, y=1.02)
plt.show()


# Save the best model
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("✅ model.pkl saved successfully!")

