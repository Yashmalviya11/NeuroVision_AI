import streamlit as st
from PIL import Image
from predict import Predictor
from utils.gradcam_utils import ViTGradCAM
from utils.database import (
    create_database,
    save_prediction,
    get_predictions,
    delete_prediction
)
from datetime import datetime
import pandas as pd
import plotly.express as px
from utils.pdf_report import generate_pdf
import json
import os
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# Login Authentication
# -----------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

USERNAME = "admin"
PASSWORD = "neurovision123"

if not st.session_state.logged_in:

    st.title("🧠 NeuroVision AI")
    st.subheader("🔐 Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login Successful!")
            st.rerun()

        else:
            st.error("Invalid Username or Password")

    st.stop()
    
with st.sidebar:
    st.success("✅ Logged in as Admin")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.rerun()
        

# -----------------------------
# Title
# -----------------------------
logo = Image.open("assets/logo.png")

col1, col2 = st.columns([1, 6])

with col1:
    st.image(logo, width=90)

with col2:
    st.title("🧠 NeuroVision AI")
    st.caption("Explainable AI Framework using Vision Transformers for Alzheimer's Disease Prediction")

st.markdown("---")

# -----------------------------
# Sidebar Navigation
# -----------------------------
page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🧠 Prediction",
        "📜 Prediction History",
        "📊 Model Performance",
        "ℹ️ About Project"
    ]
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return Predictor()

predictor = load_model()
gradcam = ViTGradCAM(predictor)

# -----------------------------
# Create SQLite Database
# -----------------------------
create_database()

# =====================================================
# Home Page
# =====================================================

if page == "🏠 Home":

    st.title("🧠 Welcome to NeuroVision AI")

    st.markdown("""
### Explainable AI Framework using Vision Transformers
for Alzheimer's Disease Prediction

NeuroVision AI is an AI-powered clinical decision support system
that predicts Alzheimer's Disease from MRI brain scans using
Vision Transformers (ViT) and explains predictions using Grad-CAM.

---

### 🚀 Features

- 🧠 Alzheimer's Disease Prediction
- 🔥 Explainable AI (Grad-CAM)
- 👤 Patient Management
- 📊 Analytics Dashboard
- 📄 PDF Medical Reports
- 📈 Model Performance
- 🗄 SQLite Database

---

### 🎯 How to Use

1. Go to **Prediction**
2. Enter Patient Details
3. Upload MRI Image
4. Click **Predict**
5. View AI Result and Grad-CAM
6. Download PDF Report

---

### 👨‍💻 Developed By

**Yash Malviya**

M.Sc. Data Science
""")

# =====================================================
# Prediction Page
# =====================================================
elif page == "🧠 Prediction":

    # -----------------------------
    # Patient Information
    # -----------------------------
    st.markdown("## 👤 Patient Information")

    patient_name = st.text_input(
        "Patient Name"
    )

    patient_id = st.text_input(
        "Patient ID"
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=60
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    st.markdown("---")

    # -----------------------------
    # Upload MRI Image
    # -----------------------------
    uploaded_file = st.file_uploader(
        "📤 Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

            image = Image.open(uploaded_file)

            col1, col2 = st.columns(2)

            # -----------------------------
            # Left Side
            # -----------------------------
            with col1:
                st.image(
                    image,
                    caption="Uploaded MRI",
                    use_container_width=True
                )

            # -----------------------------
            # Right Side
            # -----------------------------
            with col2:

                if st.button("🧠 Predict"):

                    # Validate Patient Details
                    if patient_name.strip() == "" or patient_id.strip() == "":
                        st.error("Please enter Patient Name and Patient ID.")
                        st.stop()

                    with st.spinner("Analyzing MRI..."):

                        result = predictor.predict(image)

                    st.success("Prediction Completed!")

                    # -----------------------------
                    # Save Prediction to Database
                    # -----------------------------
                    save_prediction(
                        patient_id=patient_id,
                        patient_name=patient_name,
                        age=age,
                        gender=gender,
                        prediction=result["prediction"],
                        confidence=float(result["confidence"]),
                        scan_date=datetime.now().strftime("%d-%m-%Y %H:%M")
                    )

                    st.success("✅ Patient record saved successfully.")

                    # -----------------------------
                    # Prediction
                    # -----------------------------
                    prediction = result["prediction"]
                    confidence = result["confidence"]

                    # -----------------------------
                    # Prediction Result
                    # -----------------------------
                    if prediction == "NonDemented":
                        st.success(f"🟢 Prediction: {prediction}")

                    elif prediction == "VeryMildDemented":
                        st.warning(f"🟡 Prediction: {prediction}")

                    elif prediction == "MildDemented":
                        st.warning(f"🟠 Prediction: {prediction}")

                    else:
                        st.error(f"🔴 Prediction: {prediction}")

                    st.info(f"📊 Confidence: {confidence:.2f}%")

                    st.markdown("---")

                    st.subheader("📈 Class Probabilities")

                    classes = [
                        "MildDemented",
                        "ModerateDemented",
                        "NonDemented",
                        "VeryMildDemented"
                    ]

                    prob_df = pd.DataFrame({
                        "Class": classes,
                        "Probability (%)": result["probabilities"] * 100
                    })

                    st.bar_chart(
                        prob_df.set_index("Class")
                    )

                    st.dataframe(
                        prob_df.style.format({
                            "Probability (%)": "{:.2f}"
                        }),
                        use_container_width=True
                    )
                    
                    # -----------------------------
                    # Grad-CAM
                    # -----------------------------
                    st.markdown("---")
                    st.subheader("🔥 Explainable AI (Grad-CAM)")

                    heatmap, highlighted = gradcam.generate(image)

                    col3, col4, col5 = st.columns(3)

                    # Original MRI
                    with col3:
                        st.image(
                            image,
                            caption="Original MRI",
                            use_container_width=True
                        )

                    # Grad-CAM Heatmap
                    with col4:
                        st.image(
                            heatmap,
                            caption="Grad-CAM Heatmap",
                            use_container_width=True
                        )

                    # Highlighted Region
                    with col5:
                        st.image(
                            highlighted,
                            caption="Highlighted Region",
                            use_container_width=True
                        )

                    # -----------------------------
                    # AI Explanation
                    # -----------------------------
                    st.markdown("---")
                    st.subheader("🧠 AI Explanation")

                    st.info(
                        f"""
        **Prediction:** {result['prediction']}

        **Confidence:** {result['confidence']:.2f}%

        The Vision Transformer focused on the highlighted brain regions shown in the Grad-CAM visualization.

        - 🔴 Red and Yellow regions contributed most to the prediction.
        - 🔵 Blue regions contributed less.
        - 🧠 Grad-CAM helps explain why the AI predicted this Alzheimer's stage.

        ⚠️ This prediction is generated by an AI model and should only be used as a clinical decision support tool. Final diagnosis must be made by a qualified neurologist or radiologist.
                        """
                    )
                    
                    # -----------------------------
                    # PDF Report
                    # -----------------------------
                    st.markdown("---")

                    pdf_file = f"{patient_id}_report.pdf"

                    generate_pdf(
                        filename=pdf_file,
                        patient_name=patient_name,
                        patient_id=patient_id,
                        age=age,
                        gender=gender,
                        prediction=result["prediction"],
                        confidence=result["confidence"]
                    )

                    with open(pdf_file, "rb") as pdf:

                        st.download_button(
                            label="📄 Download Medical Report",
                            data=pdf,
                            file_name=pdf_file,
                            mime="application/pdf"
                        )
                    
# =====================================================
# Prediction History
# =====================================================
elif page == "📜 Prediction History":

    st.title("📜 Prediction History")

    records = get_predictions()

    if len(records) == 0:
        st.warning("No prediction records found.")

    else:

        df = pd.DataFrame(
            records,
            columns=[
                "Patient ID",
                "Patient Name",
                "Age",
                "Gender",
                "Prediction",
                "Confidence (%)",
                "Date"
            ]
        )
                
        # -----------------------------
        # Search & Filter
        # -----------------------------

        st.subheader("🔍 Search Records")

        col1, col2 = st.columns(2)

        with col1:
            search_id = st.text_input("Search by Patient ID")

        with col2:
            search_name = st.text_input("Search by Patient Name")

        filtered_df = df.copy()

        if search_id:
            filtered_df = filtered_df[
                filtered_df["Patient ID"].astype(str).str.contains(
                    search_id,
                    case=False,
                    na=False
                )
            ]

        if search_name:
            filtered_df = filtered_df[
                filtered_df["Patient Name"].str.contains(
                    search_name,
                    case=False,
                    na=False
                )
            ]
            
        if filtered_df.empty:
            st.warning("No matching records found.")
        else:
            # Dashboard
            # Metrics
            # Pie Chart
            # Bar Chart
            # Prediction History Table
            
            # -----------------------------
            # Dashboard Metrics
            # -----------------------------

            st.subheader("📊 Dashboard")

            total_patients = len(filtered_df)

            healthy = len(
                filtered_df[
                    filtered_df["Prediction"] == "NonDemented"
                ]
            )

            demented = total_patients - healthy

            avg_confidence = filtered_df["Confidence (%)"].mean()

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("👥 Total Patients", total_patients)

            with col2:
                st.metric("🟢 Healthy", healthy)

            with col3:
                st.metric("🔴 Demented", demented)

            with col4:
                st.metric(
                    "📈 Avg Confidence",
                    f"{avg_confidence:.2f}%"
                )

            st.markdown("---")
            
            # -----------------------------
            # Prediction Distribution
            # -----------------------------

            st.subheader("📈 Prediction Distribution")

            prediction_count = (
                filtered_df["Prediction"]
                .value_counts()
                .reset_index()
            )

            prediction_count.columns = [
                "Prediction",
                "Count"
            ]

            fig = px.pie(
                prediction_count,
                values="Count",
                names="Prediction",
                hole=0.4,
                title="Prediction Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            st.markdown("---")
            
            # -----------------------------
            # Confidence by Prediction
            # -----------------------------

            st.subheader("📊 Average Confidence by Prediction")

            avg_conf = (
                filtered_df
                .groupby("Prediction")["Confidence (%)"]
                .mean()
                .reset_index()
            )

            fig2 = px.bar(
                avg_conf,
                x="Prediction",
                y="Confidence (%)",
                text_auto=".2f",
                title="Average Confidence"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )
            
            st.download_button(
                label="📥 Export History (CSV)",
                data=filtered_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv"
            )

            # -----------------------------
            # Dashboard
            # -----------------------------
            st.markdown("## 📊 Dashboard")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Patients", len(filtered_df))

            with col2:
                st.metric(
                    "Healthy",
                    len(filtered_df[filtered_df["Prediction"] == "NonDemented"])
                )

            with col3:
                st.metric(
                    "Demented",
                    len(filtered_df[filtered_df["Prediction"] != "NonDemented"])
                )

            with col4:
                avg_conf = filtered_df["Confidence (%)"].mean()

                st.metric(
                    "Avg Confidence",
                    f"{avg_conf:.2f}%" if not filtered_df.empty else "0%"
                )

            st.markdown("---")

            # -----------------------------
            # Pie Chart
            # -----------------------------
            st.subheader("📈 Prediction Distribution")

            fig = px.pie(
                filtered_df,
                names="Prediction",
                title="Prediction Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -----------------------------
            # Bar Chart
            # -----------------------------
            fig2 = px.bar(
                filtered_df,
                x="Prediction",
                color="Prediction",
                title="Prediction Count"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True
            )

            # -----------------------------
            # Prediction History Table
            # -----------------------------
            st.markdown("---")
            st.subheader("📋 Prediction History")

            st.dataframe(
                filtered_df,
                use_container_width=True
            )
            st.markdown("---")
            st.subheader("🗑 Delete Patient Record")

            delete_id = st.text_input("Enter Patient ID to Delete")

            # Session state for confirmation
            if "confirm_delete" not in st.session_state:
                st.session_state.confirm_delete = False

            if st.button("Delete Record"):

                if delete_id.strip() == "":
                    st.warning("Please enter a Patient ID.")

                else:
                    st.session_state.confirm_delete = True

            if st.session_state.confirm_delete:

                st.warning(f"⚠ Are you sure you want to delete Patient ID: {delete_id}?")

                col1, col2 = st.columns(2)

                with col1:
                    if st.button("✅ Yes, Delete"):
                        delete_prediction(delete_id)
                        st.success("Patient record deleted successfully.")
                        st.session_state.confirm_delete = False
                        st.rerun()

                with col2:
                    if st.button("❌ Cancel"):
                        st.session_state.confirm_delete = False
                        st.info("Deletion cancelled.")
                        
# =====================================================
# Model Performance
# =====================================================

elif page == "📊 Model Performance":

    st.title("📊 Model Performance")

    if not os.path.exists("metrics.json"):
        st.warning("Please run evaluate_model.py first.")
        st.stop()

    with open("metrics.json") as f:
        metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{metrics['accuracy']}%"
        )

    with col2:
        st.metric(
            "Precision",
            f"{metrics['precision']}%"
        )

    with col3:
        st.metric(
            "Recall",
            f"{metrics['recall']}%"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{metrics['f1_score']}%"
        )

    st.markdown("---")

    st.subheader("📈 Confusion Matrix")

    if os.path.exists("confusion_matrix.png"):
        st.image(
            "confusion_matrix.png",
            use_container_width=True
        )
    else:
        st.warning("Confusion Matrix not found.")

    st.markdown("---")

    st.subheader("ℹ Model Information")

    info = {
        "Model": "Vision Transformer (ViT Small)",
        "Architecture": "vit_small_patch16_224",
        "Image Size": "224 × 224",
        "Classes": "4",
        "Epochs": "20",
        "Optimizer": "AdamW",
        "Loss Function": "CrossEntropyLoss",
        "Dataset": "MRI Brain Alzheimer's Dataset"
    }

    st.table(info)
    
# =====================================================
# About Project
# =====================================================

elif page == "ℹ️ About Project":

    st.title("ℹ️ About NeuroVision AI")

    st.markdown("""
    ## 🧠 Project Overview

    **NeuroVision AI** is an Explainable Artificial Intelligence framework
    developed for the early prediction of Alzheimer's Disease using
    Vision Transformer (ViT).

    The system analyzes MRI brain scans, predicts the Alzheimer's stage,
    and explains the prediction using Grad-CAM visualization, making
    AI decisions more transparent and interpretable.
    """)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎯 Objective")

        st.write("""
- Early Alzheimer's Disease Prediction
- Assist doctors with AI-based decision support
- Improve prediction transparency using Explainable AI
- Generate automated patient reports
""")

        st.subheader("🤖 AI Model")

        st.write("""
- Vision Transformer (ViT Small)
- Transfer Learning
- PyTorch Framework
- Grad-CAM Explainability
""")

    with col2:

        st.subheader("🛠 Technology Stack")

        st.write("""
- Python
- Streamlit
- PyTorch
- TIMM
- OpenCV
- SQLite
- Plotly
- ReportLab
""")

        st.subheader("📂 Dataset")

        st.write("""
MRI Brain Alzheimer's Dataset

Classes:

• Mild Demented
• Moderate Demented
• Non Demented
• Very Mild Demented
""")

    st.markdown("---")

    st.subheader("✨ Key Features")

    st.write("""
✅ Secure Login Authentication

✅ MRI Image Upload

✅ Alzheimer's Stage Prediction

✅ Confidence Score

✅ Explainable AI (Grad-CAM)

✅ Patient Database

✅ Prediction History

✅ Dashboard & Analytics

✅ PDF Medical Report

✅ CSV Export

✅ Model Performance Evaluation
""")

    st.markdown("---")

    st.subheader("👨‍💻 Developer Information")

    st.table({
        "Field": [
            "Project Name",
            "Developer",
            "Course",
            "Technology",
            "Model",
            "Database"
        ],
        "Details": [
            "NeuroVision AI",
            "Yash Malviya",
            "M.Sc. Data Science",
            "Python + Streamlit",
            "Vision Transformer (ViT)",
            "SQLite"
        ]
    })

    st.markdown("---")

    st.success(
        "NeuroVision AI is developed for educational and research purposes only. "
        "It should assist healthcare professionals and must not replace clinical diagnosis."
    )
    
st.markdown("---")

st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 14px;'>
        🧠 <b>NeuroVision AI v1.0</b><br>
        Developed by <b>Yash Malviya</b> | M.Sc. Data Science<br>
        Vision Transformer (ViT) • Explainable AI • Streamlit
    </div>
    """,
    unsafe_allow_html=True
)