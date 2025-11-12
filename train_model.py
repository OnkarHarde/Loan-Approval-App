# train_model.py
import pandas as pd
import pymysql
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

# ========== 1. Connect to MySQL and Load Data ==========
conn = pymysql.connect(
    host="localhost",
    user="root",          # 🔧 change if needed
    password="onkar",  # 🔧 change if needed
    database="loan_db"
)

df_applicant = pd.read_sql("SELECT * FROM applicant_info", conn)
df_financial = pd.read_sql("SELECT * FROM financial_info", conn)
df_loan = pd.read_sql("SELECT * FROM loan_info", conn)

conn.close()
print("✅ Data loaded from MySQL")

# ========== 2. Merge Data ==========
df = df_applicant.merge(df_financial, on="Loan_ID", how="inner")
df = df.merge(df_loan, on="Loan_ID", how="inner")

print("Merged Data Shape:", df.shape)

# ========== 3. Preprocessing ==========
# Drop Loan_ID (not useful for prediction)
df.drop(columns=["Loan_ID"], inplace=True)

# Encode categorical variables
cat_cols = df.select_dtypes(include=["object"]).columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = df[col].astype(str)  # ensure string
    df[col] = le.fit_transform(df[col])

# Handle missing values (replace with mean for numeric, mode for categorical)
for col in df.columns:
    if df[col].dtype in [np.float64, np.int64]:
        df[col] = df[col].fillna(df[col].mean())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

# Split features and target
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

# Scale numeric values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 4. Train/Test Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ========== 5. Train Model ==========
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# ========== 6. Evaluate ==========
y_pred = model.predict(X_test)

print("\n🎯 Model Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ========== 7. Save Model with Pickle ==========
with open("loan_approval_model.pkl", "wb") as f:
    pickle.dump((model, scaler, le), f)

print("💾 Model saved as loan_approval_model.pkl")


# ================= 8. Exploratory Data Analysis (EDA) =================
print("\n📊 Exploratory Data Analysis (EDA) Questions\n")

# 1. What are the first 5 rows of the dataset?
print("\n1. First 5 rows:\n", df.head())

# 2. What are the last 5 rows of the dataset?
print("\n2. Last 5 rows:\n", df.tail())

# 3. What are the column names?
print("\n3. Column names:\n", df.columns.tolist())

# 4. What is the shape of the dataset (rows, columns)?
print("\n4. Shape of dataset:\n", df.shape)

# 5. What are the data types of each column?
print("\n5. Data types:\n", df.dtypes)

# 6. How many missing values are present in each column?
print("\n6. Missing values per column:\n", df.isnull().sum())

# 7. What is the summary statistics of numerical features?
print("\n7. Summary statistics:\n", df.describe())

# 8. What are the unique values of each categorical column?
print("\n8. Unique values in categorical columns:")
for col in df.select_dtypes(include="object").columns:
    print(f"   {col}: {df[col].unique()}")

# 9. What is the distribution of the target variable (Loan_Status)?
print("\n9. Loan_Status distribution:\n", df["Loan_Status"].value_counts())

# 10. What percentage of loans are approved vs not approved?
print("\n10. Loan_Status percentage:\n", df["Loan_Status"].value_counts(normalize=True) * 100)

# 11. What is the average applicant income?
print("\n11. Average applicant income:\n", df["ApplicantIncome"].mean())

# 12. What is the median loan amount?
print("\n12. Median loan amount:\n", df["LoanAmount"].median())

# 13. What is the maximum loan term?
print("\n13. Max loan term:\n", df["Loan_Amount_Term"].max())

# 14. What is the minimum credit history value?
print("\n14. Min credit history:\n", df["Credit_History"].min())

# 15. How many unique property areas are there?
print("\n15. Unique property areas:\n", df["Property_Area"].nunique())

# 16. Which property area has the highest number of loans?
print("\n16. Property area loan counts:\n", df["Property_Area"].value_counts())

# 17. Which education group has more applicants?
print("\n17. Education counts:\n", df["Education"].value_counts())

# 18. How many applicants are self-employed?
print("\n18. Self employed count:\n", df["Self_Employed"].value_counts())

# 19. What is the average loan amount for approved vs non-approved loans?
print("\n19. Avg LoanAmount by Loan_Status:\n", df.groupby("Loan_Status")["LoanAmount"].mean())

# 20. What is the average applicant income for graduates vs non-graduates?
print("\n20. Avg ApplicantIncome by Education:\n", df.groupby("Education")["ApplicantIncome"].mean())

# 21. Which gender group has higher average loan amount?
print("\n21. Avg LoanAmount by Gender:\n", df.groupby("Gender")["LoanAmount"].mean())

# 22. What is the most common loan amount term?
print("\n22. Most common loan term:\n", df["Loan_Amount_Term"].mode()[0])

# 23. What is the distribution of number of dependents?
print("\n23. Dependents distribution:\n", df["Dependents"].value_counts())

# 24. How many applicants have income greater than 10,000?
print("\n24. Applicants with income > 10,000:\n", (df["ApplicantIncome"] > 10000).sum())

# 25. What is the maximum applicant income in the dataset?
print("\n25. Max applicant income:\n", df["ApplicantIncome"].max())

# 26. What is the minimum loan amount in the dataset?
print("\n26. Min loan amount:\n", df["LoanAmount"].min())

# 27. Which marital status group has higher loan approval rate?
print("\n27. Loan approval rate by Married:\n", df.groupby("Married")["Loan_Status"].mean())

# 28. Which property area has highest loan approval rate?
print("\n28. Loan approval rate by Property_Area:\n", df.groupby("Property_Area")["Loan_Status"].mean())

# 29. What is the average coapplicant income among approved loans?
print("\n29. Avg CoapplicantIncome (approved):\n", df[df["Loan_Status"]==1]["CoapplicantIncome"].mean())

# 30. Which numeric column has the highest correlation with Loan_Status?
print("\n30. Correlation with Loan_Status:\n", df.corr()["Loan_Status"].drop("Loan_Status").sort_values(ascending=False).head(1))

