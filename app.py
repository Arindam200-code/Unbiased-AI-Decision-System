import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_data
from model import train_model
from fairness import check_bias

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Unbiased AI System", layout="wide")
st.title("⚖️ Unbiased AI Decision System")

domain = st.selectbox(
    "Select Domain",
    ["Job Hiring", "Education", "Medical Care", "Bank Loan"]
)


# =========================
# SESSION STATE INIT
# =========================
for key in ["df", "model", "X_test", "y_test", "predictions", "accuracy", "report", "fairness_score"]:
    if key not in st.session_state:
        st.session_state[key] = None


# =========================
# PDF REPORT
# =========================
def generate_pdf(accuracy, fairness_score, domain):

    file_name = "AI_Fairness_Report.pdf"
    doc = SimpleDocTemplate(file_name)

    styles = getSampleStyleSheet()
    content = []

    accuracy = accuracy or 0
    fairness_score = fairness_score or 0

    content.append(Paragraph(f"{domain} AI Fairness Report", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Model Accuracy: {accuracy:.2f}", styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Bias Score: {fairness_score:.3f}", styles['Normal']))
    content.append(Spacer(1, 12))

    verdict = "Bias Detected" if abs(fairness_score) > 0.1 else "Model Appears Fair"
    content.append(Paragraph(f"Conclusion: {verdict}", styles['Normal']))

    doc.build(content)
    return file_name


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Data", "⚙️ Configure", "🔍 Analyse", "📈 Result", "📄 Report"]
)


# =========================
# 📊 DATA
# =========================
with tab1:

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("Dataset Loaded")
        st.dataframe(st.session_state.df.head())


# =========================
# ⚙️ TRAIN
# =========================
with tab2:

    if st.session_state.df is not None:

        X, y = preprocess_data(st.session_state.df)

        model, X_test, y_test, predictions, accuracy, report = train_model(X, y)

        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.predictions = predictions
        st.session_state.accuracy = accuracy
        st.session_state.report = report

        st.success("Model Trained Successfully")

    else:
        st.warning("Upload dataset first")


# =========================
# 🔍 ANALYSE
# =========================
with tab3:

    if st.session_state.df is not None:

        df = st.session_state.df
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        predictions = st.session_state.predictions

        # SAFE ALIGNMENT
        gender = df.loc[X_test.index, "gender"]

        # FAIRNESS
        fairness_score = check_bias(y_test, predictions, gender)
        st.session_state.fairness_score = fairness_score

        st.metric("Accuracy", f"{st.session_state.accuracy:.2f}")
        st.metric("Bias Score", f"{fairness_score:.3f}")

        st.text(st.session_state.report)

        # Gender distribution
        st.subheader("Gender Distribution")
        st.bar_chart(df["gender"].value_counts())

        # Selection rate
        st.subheader("Selection Rate by Gender")

        temp = df.loc[X_test.index].copy()
        temp["prediction"] = predictions

        st.bar_chart(temp.groupby("gender")["prediction"].mean())

        # Feature importance
        st.subheader("Feature Importance")

        fig, ax = plt.subplots()
        ax.barh(X_test.columns, st.session_state.model.feature_importances_)
        st.pyplot(fig)

    else:
        st.warning("Upload dataset first")


# =========================
# 📈 RESULT
# =========================
with tab4:

    if st.session_state.fairness_score is not None:

        if abs(st.session_state.fairness_score) > 0.1:
            st.error("⚠️ Bias Detected")
        else:
            st.success("✅ Model Appears Fair")

        st.write(f"Accuracy: {st.session_state.accuracy:.2f}")
        st.write(f"Bias Score: {st.session_state.fairness_score:.3f}")

    else:
        st.warning("Run analysis first")


# =========================
# 📄 REPORT
# =========================
with tab5:

    if st.button("Generate PDF"):

        pdf = generate_pdf(
            st.session_state.accuracy,
            st.session_state.fairness_score,
            domain
        )

        with open(pdf, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="AI_Fairness_Report.pdf"
            )

    else:
        st.warning("Run analysis first")