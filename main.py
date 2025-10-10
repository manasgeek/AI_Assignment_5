import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, f_classif

# Streamlit Page Config
st.set_page_config(page_title="Addiction Prediction ML App", layout="wide")
st.title("Addiction Prediction ML Application")

# -------------------- Upload Dataset --------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Columns to exclude from feature selection
exclude_features = {
    "location",
    "Daily usage hours",
    "sleep hours",
    "time on education",
    "weekend usage hour",
    "addiction level"
}

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Dataset")
    st.dataframe(df.head())
    st.write(f"Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    # -------------------- Data Cleaning --------------------
    drop_cols = [col for col in df.columns if "id" in col.lower() or "name" in col.lower()]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    thresh = len(df) * 0.5
    df.dropna(axis=1, thresh=thresh, inplace=True)

    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    st.subheader("Cleaned Dataset")
    st.dataframe(df.head())

    # -------------------- Encoding --------------------
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -------------------- EDA --------------------
    st.subheader("Exploratory Data Analysis")

    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        st.write("Histograms of numeric features")
        col1, col2 = st.columns(2)
        for i, col in enumerate(numeric_cols[:4]):
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            if i % 2 == 0:
                col1.pyplot(fig)
            else:
                col2.pyplot(fig)

    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
    st.pyplot(fig)

    # -------------------- Feature Selection --------------------
    st.sidebar.header("Feature Selection")
    target_col = st.sidebar.selectbox("Select Target Column", df.columns)

    feature_cols = [col for col in df.columns if col != target_col and col not in exclude_features]

    X = df[feature_cols]
    y = df[target_col]

    k = min(6, len(feature_cols))
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)
    top_features = X.columns[selector.get_support()].tolist()

    st.write("Top Selected Features:", top_features)

    X = X[top_features]

    # -------------------- Scaling --------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -------------------- Train Test Split --------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------- Model Training --------------------
    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Random Forest": RandomForestClassifier(),
        "SVM": SVC()
    }

    st.subheader("Model Performance")

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        results.append([name, acc, prec, rec, f1])

    results_df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score"])
    st.dataframe(results_df)

    best_model_name = results_df.iloc[results_df['Accuracy'].idxmax()]['Model']
    best_model = models[best_model_name]
    st.success(f"Best Model Selected: {best_model_name}")

    # -------------------- User Input for Prediction --------------------
    st.subheader("Prediction for New Input")

    user_inputs = []
    for col in top_features:
        if col in numeric_cols:
            val = st.number_input(
                f"{col}",
                float(df[col].min()),
                float(df[col].max()),
                float(df[col].mean())
            )
            user_inputs.append(val)
        else:
            if col in label_encoders:
                options = list(label_encoders[col].classes_)
                val = st.selectbox(f"{col}", options=options)
                val = label_encoders[col].transform([val])[0]
                user_inputs.append(val)
            else:
                val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
                user_inputs.append(val)

    if st.button("Predict"):
        user_scaled = scaler.transform([user_inputs])
        prediction = best_model.predict(user_scaled)[0]
        label = "Addicted" if prediction == 1 else "Not Addicted"
        st.write(f"Prediction: {label}")

else:
    st.info("Upload a CSV file to start the process.")
