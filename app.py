# ==========================================================
# Consumer Complaint Analyzer
# Part 1 - Foundation
# ==========================================================

# ===========================
# Import Libraries
# ===========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import plotly.express as px

# ===========================
# Website Configuration
# ===========================

st.set_page_config(
    page_title="Consumer Complaint Analyzer",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================
# Custom CSS
# ===========================

st.markdown("""
<style>

.main{
    background-color:#F8FAFC;
}

h1{
    color:#1E3A8A;
}

h2{
    color:#2563EB;
}

h3{
    color:#2563EB;
}

.stButton>button{
    background-color:#2563EB;
    color:white;
    border-radius:10px;
    height:50px;
    width:200px;
    font-size:18px;
    border:none;
}

.stButton>button:hover{
    background-color:#1D4ED8;
    color:white;
}

.css-1d391kg{
    background-color:#F1F5F9;
}

</style>
""", unsafe_allow_html=True)

# ===========================
# Load Model
# ===========================

@st.cache_resource
def load_model():

    model = joblib.load("complaint_classifier.pkl")

    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()

# ===========================
# Clean Text Function
# ===========================

def clean_text(text):

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = " ".join(text.split())

    return text


# ===========================
# Prediction Function
# ===========================

def predict_category(text):

    cleaned = clean_text(text)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)

    return prediction[0]


# ===========================
# Sidebar
# ===========================

st.sidebar.title("🤖 Consumer Complaint Analyzer")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "🤖 Predict",

        "📂 Bulk Prediction",

        "📊 Dashboard",

        "ℹ️ About"

    ],

    key="main_navigation"

)

# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🤖 Consumer Complaint Analyzer")

    st.markdown("---")

    st.subheader("Welcome")

    st.write("""

This AI application classifies consumer complaints into financial product categories using Machine Learning and Natural Language Processing.

The model has been trained using the Consumer Financial Protection Bureau (CFPB) Complaint Dataset.

""")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Features")

        st.success("✔ Complaint Classification")

        st.success("✔ NLP Text Cleaning")

        st.success("✔ TF-IDF Vectorization")

        st.success("✔ Machine Learning Prediction")

    with col2:

        st.subheader("Technology Stack")

        st.info("Python")

        st.info("Pandas")

        st.info("Scikit-Learn")

        st.info("Streamlit")

    st.markdown("---")

    st.subheader("How It Works")

    st.write("""

1️⃣ User enters a complaint.

2️⃣ Complaint text is cleaned.

3️⃣ TF-IDF converts text into numerical features.

4️⃣ Machine Learning predicts the complaint category.

5️⃣ Result is displayed instantly.

""")

    st.success("✅ Model Loaded Successfully")

# ==========================================================
# Remaining pages will be added in Parts 2–5
# ==========================================================













# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🤖 Predict":

    st.title("🤖 Consumer Complaint Prediction")

    st.markdown("---")

    st.write(
        """
Enter your consumer complaint below.

The AI model will analyze the complaint and predict the most likely financial product category.
"""
    )

    # -----------------------------
    # Example Complaints
    # -----------------------------

    st.subheader("📌 Example Complaints")

    col1, col2 = st.columns(2)

    with col1:

        if st.button("Credit Card Example"):

            st.session_state.example = (
                "My credit card was charged twice "
                "and the bank is refusing to refund me."
            )

    with col2:

        if st.button("Loan Example"):

            st.session_state.example = (
                "The bank rejected my loan application "
                "without giving any reason."
            )

    if "example" not in st.session_state:

        st.session_state.example = ""

    # -----------------------------
    # Complaint Input
    # -----------------------------

    complaint = st.text_area(

        "Enter Complaint",

        value=st.session_state.example,

        height=220

    )

    st.markdown("---")

    # -----------------------------
    # Prediction Button
    # -----------------------------

    if st.button("🔍 Predict Category"):

        if complaint.strip() == "":

            st.warning("Please enter a complaint.")

        else:

            with st.spinner("Analyzing Complaint..."):

                category = predict_category(complaint)

            st.success("Prediction Completed!")

            st.markdown("## 📊 Prediction Result")

            st.metric(

                label="Predicted Category",

                value=category

            )

            st.markdown("---")

            st.subheader("📝 Cleaned Complaint")

            st.code(

                clean_text(complaint),

                language="text"

            )

            st.markdown("---")

            st.subheader("💡 AI Interpretation")

            st.info(

                f"""
The complaint is most likely related to:

**{category}**

The prediction was generated using your trained
Machine Learning model and TF-IDF vectorizer.
"""
            )

    st.markdown("---")

    st.subheader("📖 Tips")

    st.write("""
✔ Enter complete complaint text.

✔ More detailed complaints generally produce better predictions.

✔ Avoid entering only one or two words.

✔ The model works best on English complaints.
""")
    













# ==========================================================
# BULK CSV PREDICTION
# ==========================================================

elif page == "📂 Bulk Prediction":

    st.title("📂 Bulk Complaint Prediction")

    st.markdown("---")

    st.write("""
Upload a CSV file containing consumer complaints.

The application will predict the complaint category for every row.
""")

    uploaded_file = st.file_uploader(

        "Choose CSV File",

        type=["csv"]

    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.success("✅ File Uploaded Successfully")

            st.subheader("Dataset Preview")

            st.dataframe(df.head())

            st.markdown("---")

            st.subheader("Select Complaint Column")

            complaint_column = st.selectbox(

                "Complaint Column",

                df.columns

            )

            if st.button("🚀 Start Prediction"):

                progress_bar = st.progress(0)

                status = st.empty()

                predictions = []

                total = len(df)

                for index, complaint in enumerate(df[complaint_column]):

                    complaint = clean_text(str(complaint))

                    vector = vectorizer.transform([complaint])

                    prediction = model.predict(vector)[0]

                    predictions.append(prediction)

                    progress = int((index + 1) / total * 100)

                    progress_bar.progress(progress)

                    status.text(

                        f"Processing {index+1} of {total} complaints..."

                    )

                df["Predicted Category"] = predictions

                status.empty()

                progress_bar.empty()

                st.success("🎉 Prediction Completed Successfully!")

                st.markdown("---")

                st.subheader("Prediction Preview")

                st.dataframe(df.head(20))

                st.markdown("---")

                st.subheader("Prediction Summary")

                summary = (

                    df["Predicted Category"]

                    .value_counts()

                    .reset_index()

                )

                summary.columns = [

                    "Category",

                    "Count"

                ]

                st.dataframe(summary)

                st.bar_chart(

                    summary.set_index("Category")

                )

                csv = df.to_csv(

                    index=False

                ).encode("utf-8")

                st.download_button(

                    label="📥 Download Predictions",

                    data=csv,

                    file_name="Predicted_Complaints.csv",

                    mime="text/csv"

                )

        except Exception as e:

            st.error(f"Error : {e}")

    else:

        st.info("Upload a CSV file to begin.")









# ==========================================================
# DASHBOARD
# ==========================================================

elif page == "📊 Dashboard":

    st.title("📊 Consumer Complaint Dashboard")

    st.markdown("---")

    st.write(
        """
This dashboard provides insights into the uploaded complaint
dataset and your trained Machine Learning model.
"""
    )

    # -------------------------------------------------
    # Model Information
    # -------------------------------------------------

    st.subheader("🤖 Model Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Algorithm",
            "Machine Learning"
        )

    with col2:
        st.metric(
            "Vectorizer",
            "TF-IDF"
        )

    with col3:
        st.metric(
            "Language",
            "English"
        )

    st.markdown("---")

    # -------------------------------------------------
    # Dataset Available?
    # -------------------------------------------------

    if "df" in locals():

        st.subheader("Dataset Statistics")

        c1, c2, c3 = st.columns(3)

        with c1:

            st.metric(
                "Rows",
                len(df)
            )

        with c2:

            st.metric(
                "Columns",
                len(df.columns)
            )

        with c3:

            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        st.markdown("---")

        # ---------------------------------------------
        # Prediction Distribution
        # ---------------------------------------------

        if "Predicted Category" in df.columns:

            counts = (
                df["Predicted Category"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Category",
                "Count"
            ]

            st.subheader("📈 Prediction Distribution")

            fig = px.bar(

                counts,

                x="Category",

                y="Count",

                text="Count",

                title="Predicted Categories"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

            st.markdown("---")

            st.subheader("🥧 Category Share")

            fig2 = px.pie(

                counts,

                values="Count",

                names="Category",

                hole=0.45

            )

            st.plotly_chart(

                fig2,

                use_container_width=True

            )

            st.markdown("---")

            st.subheader("Top Categories")

            st.dataframe(counts)

        else:

            st.warning(

                "Run Bulk Prediction first to generate dashboard charts."

            )

    else:

        st.info(

            "Upload a CSV file in the Bulk Prediction page to view analytics."

        )

    st.markdown("---")

    # -------------------------------------------------
    # Application Features
    # -------------------------------------------------

    st.subheader("🚀 Features")

    feature_df = pd.DataFrame({

        "Feature":[

            "Complaint Prediction",

            "CSV Upload",

            "Bulk Prediction",

            "Download CSV",

            "Interactive Dashboard",

            "Machine Learning"

        ],

        "Status":[

            "✅",

            "✅",

            "✅",

            "✅",

            "✅",

            "✅"

        ]

    })

    st.dataframe(feature_df)

    st.markdown("---")

    st.success("Dashboard Loaded Successfully!")











# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About Consumer Complaint Analyzer")

    st.markdown("---")

    st.header("📌 Project Overview")

    st.write("""
The **Consumer Complaint Analyzer** is an Artificial Intelligence application
developed using **Natural Language Processing (NLP)** and **Machine Learning**.

The application predicts the financial product category of a consumer complaint
based on the complaint text.

This project was developed as a Data Science portfolio project.
""")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    tech = pd.DataFrame({

        "Technology":[

            "Python",

            "Pandas",

            "NumPy",

            "Scikit-Learn",

            "Streamlit",

            "Joblib",

            "TF-IDF",

            "Machine Learning"

        ]

    })

    st.table(tech)

    st.markdown("---")

    st.header("✨ Features")

    st.success("✔ Single Complaint Prediction")

    st.success("✔ Bulk CSV Prediction")

    st.success("✔ Interactive Dashboard")

    st.success("✔ CSV Download")

    st.success("✔ Machine Learning")

    st.success("✔ NLP Text Processing")

    st.markdown("---")

    st.header("📂 Dataset")

    st.info("""
Consumer Financial Protection Bureau (CFPB)

This dataset contains thousands of consumer complaints related to financial products.
""")

    st.markdown("---")

    st.header("📊 Machine Learning Workflow")

    workflow = """

Dataset

↓

Data Cleaning

↓

Text Preprocessing

↓

TF-IDF Vectorization

↓

Machine Learning Model

↓

Prediction

↓

Website Deployment

"""

    st.code(workflow)

    st.markdown("---")

    st.header("👨‍💻 Developer")

    st.write("""

**Name:** Atharv Gupta

BS-MS Mathematics & Data Science

Harcourt Butler Technical University (HBTU), Kanpur

""")

    st.markdown("---")

    st.header("🚀 Future Improvements")

    st.write("""

• Deep Learning Models

• BERT / RoBERTa

• LLM Integration

• Complaint Summarization

• Auto Response Generation

• Legal Recommendation System

• Multi-language Support

""")

    st.markdown("---")

    st.success("Thank you for using Consumer Complaint Analyzer!")








