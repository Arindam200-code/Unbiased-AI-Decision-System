import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess_data
from model import train_model
from fairness import check_bias

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="AI Fairness System", layout="wide")

st.title("⚖️ AI Fair Decision System")

# =========================
# DOMAIN SELECTION (SECOND LEVEL MENU)
# =========================
domain = st.selectbox(
    "Select Domain",
    ["Job Hiring", "Education", "Medical Care", "Bank Loan"]
)


# =========================
# PDF REPORT
# =========================
def generate_pdf(accuracy, fairness_score, domain):

    file_name = "AI_Fairness_Report.pdf"
    doc = SimpleDocTemplate(file_name)

    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph(f"{domain} AI Fairness Report", styles['Title']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Model Accuracy: {accuracy:.2f}", styles['Normal']))
    content.append(Spacer(1, 12))

    content.append(Paragraph(f"Fairness Score: {fairness_score:.2f}", styles['Normal']))
    content.append(Spacer(1, 12))

    verdict = "Bias Detected" if abs(fairness_score) > 0.1 else "Model Appears Fair"
    content.append(Paragraph(f"Conclusion: {verdict}", styles['Normal']))

    doc.build(content)
    return file_name


# =========================
# TABS (FIRST LEVEL NAVIGATION)
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["📊 Data", "⚙️ Configure", "🔍 Analyse", "📈 Result", "📄 Report"]
)


# =========================
# GLOBAL VARIABLES
# =========================
df = None
X = y = gender = None
model = predictions = y_test = X_test = accuracy = report = None
fairness_score = None


# =========================
# 📊 DATA TAB (CSV UPLOAD ONLY HERE)
# =========================
with tab1:
    st.subheader(f"{domain} Dataset Upload")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset Loaded Successfully")
        st.dataframe(df.head())

        X, y, gender = preprocess_data(df)


# =========================
# ⚙️ CONFIGURE TAB
# =========================
with tab2:
    st.subheader("Model Configuration")

    st.write("Model: Random Forest")
    st.write("Test Size: 20%")
    st.write("Random State: 42")

    if df is not None:
        model, predictions, y_test, X_test, accuracy, report = train_model(X, y)
        st.success("Model Trained Successfully")
    else:
        st.warning("Please upload dataset in Data tab first")


# =========================
# 🔍 ANALYSE TAB (GRAPHS)
# =========================
with tab3:

    if df is not None:

        sensitive_feature = gender.iloc[X_test.index]
        fairness_score = check_bias(y_test, predictions, sensitive_feature)

        st.subheader("Model Analysis")

        st.metric("Accuracy", f"{accuracy:.2f}")
        st.metric("Fairness Score", f"{fairness_score:.2f}")

        st.text(report)

        # -------------------------
        # Gender Distribution
        # -------------------------
        st.subheader("Gender Distribution")

        gender_counts = df['gender'].value_counts()

        fig1, ax1 = plt.subplots()
        ax1.bar(gender_counts.index, gender_counts.values)
        ax1.set_title("Gender Distribution")
        st.pyplot(fig1)

        # -------------------------
        # Bias Graph
        # -------------------------
        st.subheader("Selection Rate by Gender")

        result_df = df.loc[X_test.index].copy()
        result_df['prediction'] = predictions

        selection_rate = result_df.groupby('gender')['prediction'].mean()

        fig2, ax2 = plt.subplots()
        ax2.bar(selection_rate.index, selection_rate.values)
        ax2.set_title("Bias Detection")
        st.pyplot(fig2)

        # -------------------------
        # Feature Importance
        # -------------------------
        st.subheader("Feature Importance")

        importance = model.feature_importances_
        features = X.columns

        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values("Importance", ascending=True)

        fig3, ax3 = plt.subplots()

        bars = ax3.barh(feat_df["Feature"], feat_df["Importance"])

        ax3.set_title("Feature Importance")

        for bar in bars:
            val = bar.get_width()
            ax3.text(val + 0.005,
                     bar.get_y() + bar.get_height()/2,
                     f"{val:.2f}",
                     va='center')

        st.pyplot(fig3)

    else:
        st.warning("Upload dataset first")


# =========================
# 📈 RESULT TAB
# =========================
with tab4:

    if df is not None and fairness_score is not None:

        st.subheader("Final Result")

        if abs(fairness_score) > 0.1:
            st.error("⚠️ Bias Detected in Model")
        else:
            st.success("✅ Model Appears Fair")

        st.write(f"Accuracy: {accuracy:.2f}")
        st.write(f"Fairness Score: {fairness_score:.2f}")

    else:
        st.warning("Run analysis first")


# =========================
# 📄 REPORT TAB
# =========================
with tab5:

    st.subheader("Download Report")

    if st.button("Generate PDF Report") and df is not None:

        pdf_file = generate_pdf(accuracy, fairness_score, domain)

        with open(pdf_file, "rb") as f:
            st.download_button(
                "⬇️ Download PDF",
                f,
                file_name="AI_Fairness_Report.pdf"
            )

    else:
        st.warning("Complete analysis first")