import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =====================================================================
# PAGE CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Teen Smartphone Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# ENHANCED STYLING
# =====================================================================
st.markdown("""
    <style>
    /* General Body and Font */
    body, .stApp {
        background-color: #0f172a; /* Dark blue-gray background */
        color: #e2e8f0; /* Light gray text */
        font-family: "Segoe UI", sans-serif;
    }

    /* Titles and Headers */
    h1 {
        text-align: center;
        color: #60a5fa; /* Light blue */
        font-weight: 700;
        padding-bottom: 10px;
    }
    h2 {
        color: #cbd5e1; /* Lighter gray for subheaders */
        border-bottom: 1px solid #334155; /* Slate border */
        padding-bottom: 8px;
        margin-top: 30px;
    }
    h3 {
        color: #93c5fd; /* Softer blue for smaller headers */
        margin-top: 15px;
    }

    /* Custom Containers and Cards */
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.5);
    }
    .st-emotion-cache-1r4qj8v { /* Main container styling */
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e293b;
        padding: 10px 20px;
        border-radius: 8px;
        font-weight: 600;
        color: #e2e8f0;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb; /* Bright blue for selected tab */
        color: white;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border: none;
        padding: 10px 24px;
        border-radius: 8px;
        cursor: pointer;
        width: 100%;
    }
    div.stButton > button:hover {
        background-color: #1e40af; /* Darker blue on hover */
    }

    /* Expanders */
    .st-emotion-cache-p5msec {
        background-color: #1e293b;
        border-radius: 8px;
    }

    </style>
""", unsafe_allow_html=True)


# =====================================================================
# HEADER
# =====================================================================
st.markdown("<h1>📱 Teen Smartphone Addiction Predictor</h1>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; padding: 8px; background-color: #1e293b; border-radius: 8px; margin-bottom: 25px;'>
    <p style='font-size: 16px; color: #93c5fd; margin: 0;'>
        Analyze teen smartphone usage patterns & predict addiction levels using Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# FUNCTIONS (No changes needed here)
# =====================================================================
@st.cache_data
def load_data(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def identify_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    return numeric_cols, categorical_cols

def calculate_statistics(df, numeric_cols):
    stats_dict = {}
    for col in numeric_cols:
        data = df[col].dropna()
        stats_dict[col] = {
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data)
        }
    return pd.DataFrame(stats_dict).T

def preprocess_data(df, target_col='Addiction_Level'):
    if target_col not in df.columns:
        st.error(f"Target column '{target_col}' not found!")
        return None, None, None, None
    cols_to_drop = [c for c in ['ID', 'Name'] if c in df.columns]
    X = df.drop(columns=cols_to_drop + [target_col])
    # Binarize target: 1 if Addiction_Level is high (>=8), else 0
    y = (df[target_col] >= 8).astype(int)
    numeric_cols, cat_cols = identify_column_types(X)
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    return X, y, label_encoders, cat_cols

@st.cache_resource
def train_models(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True, random_state=42),
        'Naive Bayes': GaussianNB()
    }
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        results[name] = {
            'model': model,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'y_pred_proba': y_pred_proba
        }
    return results, scaler

# =====================================================================
# SIDEBAR UPLOAD
# =====================================================================
st.sidebar.header("📂 Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV data",
    type=['csv'],
    help="Ensure the CSV contains features like usage hours, age, gender, etc., and an 'Addiction_Level' column."
)

if uploaded_file is None:
    st.info("👈 Upload a CSV file to begin analysis.")
    st.markdown("""
    ### Expected Columns:
    - **Demographics**: `Age`, `Gender`, `School_Grade`, `Location`
    - **Usage Metrics**: `Daily_Usage_Hours`, `Weekend_Usage_Hours`, `Phone_Checks_Per_Day`
    - **Well-being**: `Sleep_Hours`, `Academic_Performance`
    - **Psychological**: `Anxiety_Level`, `Depression_Level`
    - **Target**: `Addiction_Level` (numeric, e.g., 1-10)
    """)
else:
    df = load_data(uploaded_file)
    if df is not None:
        numeric_cols, cat_cols = identify_column_types(df.drop(columns=['ID', 'Name'], errors='ignore'))

        tab_list = ["🏠 Home", "📊 EDA", "📈 Statistics", "🤖 Model Performance", "🎯 Prediction"]
        tabs = st.tabs(tab_list)

        # HOME TAB
        with tabs[0]:
            st.header("Dataset Overview")
            col1, col2 = st.columns(2)
            col1.markdown(f"<div class='metric-card'><h3>Rows</h3><h2>{len(df)}</h2></div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric-card'><h3>Columns</h3><h2>{len(df.columns)}</h2></div>", unsafe_allow_html=True)

            with st.container():
                st.subheader("Data Preview")
                st.dataframe(df.head(10), use_container_width=True)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Numeric Columns")
                st.expander("View Numeric Columns", expanded=False).write(numeric_cols)
            with col2:
                st.subheader("Categorical Columns")
                st.expander("View Categorical Columns", expanded=False).write(cat_cols)


        # EDA TAB
        with tabs[1]:
            st.header("Exploratory Data Analysis")
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Numeric Feature Distribution")
                if numeric_cols:
                    selected_num = st.selectbox("Select a numeric feature:", numeric_cols, key="num_select")
                    fig = px.histogram(df, x=selected_num, nbins=30, template="plotly_dark", title=f"Distribution of {selected_num}")
                    st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Categorical Feature Distribution")
                if cat_cols:
                    selected_cat = st.selectbox("Select a categorical feature:", cat_cols, key="cat_select")
                    fig = px.bar(df[selected_cat].value_counts(), template="plotly_dark", title=f"Counts of {selected_cat}")
                    fig.update_layout(xaxis_title=selected_cat, yaxis_title="Count", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Correlation Heatmap")
            if len(numeric_cols) > 1:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r', template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)


        # STATISTICS TAB
        with tabs[2]:
            st.header("Descriptive Statistics")
            stats_df = calculate_statistics(df, numeric_cols)
            if not stats_df.empty:
                selected_stat_col = st.selectbox("Select column to view key stats:", numeric_cols, key="stat_select")
                s = stats_df.loc[selected_stat_col]
                cols = st.columns(4)
                cols[0].metric("Mean", f"{s['Mean']:.2f}")
                cols[1].metric("Median", f"{s['Median']:.2f}")
                cols[2].metric("Std Dev", f"{s['Std Dev']:.2f}")
                cols[3].metric("Skewness", f"{s['Skewness']:.2f}")
                
                with st.expander("Show Full Statistics Table"):
                    st.dataframe(stats_df, use_container_width=True)


        # MODEL PERFORMANCE TAB
        with tabs[3]:
            st.header("Model Training & Performance")
            with st.spinner("Training models... This might take a moment. ⏳"):
                X, y, label_encoders, cat_cols_processed = preprocess_data(df)
                if X is not None:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    results, scaler = train_models(X_train, X_test, y_train, y_test)
                    
                    # Store results in session state for other tabs
                    st.session_state['models'] = results
                    st.session_state['scaler'] = scaler
                    st.session_state['label_encoders'] = label_encoders
                    st.session_state['feature_names'] = X.columns.tolist()
                    st.session_state['cat_cols'] = cat_cols_processed
                    st.session_state['y_test'] = y_test

            if 'models' in st.session_state:
                st.subheader("Performance Metrics")
                perf_data = []
                for m, r in st.session_state['models'].items():
                    perf_data.append({
                        'Model': m,
                        'Accuracy': r['accuracy'], 'Precision': r['precision'],
                        'Recall': r['recall'], 'F1-Score': r['f1_score'], 'ROC-AUC': r['roc_auc']
                    })
                perf_df = pd.DataFrame(perf_data).set_index('Model')
                st.dataframe(perf_df.style.format("{:.3f}").background_gradient(cmap='Blues'), use_container_width=True)

                st.subheader("Visual Model Comparison")
                metric_to_plot = st.selectbox("Select metric to compare", ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
                sorted_perf = perf_df[metric_to_plot].sort_values(ascending=False)
                fig = px.bar(sorted_perf, x=sorted_perf.index, y=metric_to_plot,
                             template="plotly_dark", title=f"Model Comparison by {metric_to_plot}",
                             text_auto='.3f', color=sorted_perf.index)
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Combined ROC Curves")
                fig = go.Figure()
                fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
                for model_name, r in st.session_state['models'].items():
                    fpr, tpr, _ = roc_curve(y_test, r['y_pred_proba'])
                    fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{model_name} (AUC={r['roc_auc']:.3f})", mode='lines'))
                
                fig.update_layout(
                    title="ROC Curve Comparison", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                    template="plotly_dark", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)


        # PREDICTION TAB
        with tabs[4]:
            st.header("Make a Prediction")
            if 'models' not in st.session_state:
                st.warning("⚠️ Models have not been trained yet. Please visit the 'Model Performance' tab first.")
            else:
                model_choice = st.selectbox("Select Model for Prediction", list(st.session_state['models'].keys()))
                feature_names = st.session_state['feature_names']
                input_data = {}

                # Create structured input fields
                with st.container():
                    st.markdown("### Demographics")
                    col1, col2 = st.columns(2)
                    demographic_features = ['Age', 'Gender', 'School_Grade', 'Location']
                    for i, feat in enumerate(d for d in demographic_features if d in feature_names):
                        col = col1 if i % 2 == 0 else col2
                        if feat in cat_cols:
                            input_data[feat] = col.selectbox(feat, df[feat].unique(), key=f"pred_{feat}")
                        else:
                            min_val, max_val, mean_val = float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())
                            input_data[feat] = col.slider(feat, min_val, max_val, value=mean_val, key=f"pred_{feat}")
                
                with st.container():
                    st.markdown("### Usage Patterns & Well-being")
                    col1, col2 = st.columns(2)
                    other_features = [f for f in feature_names if f not in demographic_features]
                    for i, feat in enumerate(other_features):
                        col = col1 if i % 2 == 0 else col2
                        if feat in cat_cols:
                            input_data[feat] = col.selectbox(feat, df[feat].unique(), key=f"pred_{feat}")
                        else:
                            min_val, max_val, mean_val = float(df[feat].min()), float(df[feat].max()), float(df[feat].mean())
                            input_data[feat] = col.slider(feat, min_val, max_val, value=mean_val, key=f"pred_{feat}")

                if st.button("🔮 Predict Addiction Risk"):
                    input_df = pd.DataFrame([input_data])
                    
                    # Preprocess input data
                    for c in st.session_state['cat_cols']:
                        if c in input_df.columns:
                            le = st.session_state['label_encoders'][c]
                            input_df[c] = le.transform(input_df[c].astype(str))
                    
                    scaled_input = st.session_state['scaler'].transform(input_df)
                    model = st.session_state['models'][model_choice]['model']
                    pred = model.predict(scaled_input)[0]
                    prob = model.predict_proba(scaled_input)[0]
                    
                    st.subheader("🎯 Prediction Result")
                    res_col1, res_col2 = st.columns(2)
                    if pred == 1:
                        res_col1.error("**High Risk** of Smartphone Addiction Detected")
                        res_col2.metric("Confidence Score", f"{prob[1]:.2%}")
                    else:
                        res_col1.success("**Low Risk** of Smartphone Addiction Detected")
                        res_col2.metric("Confidence Score", f"{prob[0]:.2%}")