import streamlit as st
st.set_page_config(page_title="Unified Cybersecurity Suite", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import asyncio
import requests
import re
from concurrent.futures import ThreadPoolExecutor
import imaplib
import email

# ML & Preprocessing libraries
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# For OCR (processing images)
from PIL import Image
import pytesseract

# -------------------------------
# IMPORTANT DEPENDENCIES:
#   pip install streamlit pandas numpy matplotlib seaborn plotly imbalanced-learn
#   pip install scikit-learn xgboost requests Pillow pytesseract
# Also install Tesseract OCR on your system (apt-get install tesseract-ocr, or from https://github.com/UB-Mannheim/tesseract)
# Enable IMAP in your Gmail account if you wish to use the Gmail Inbox feature.
# -------------------------------

# -------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------
YOUR_VIRUSTOTAL_API_KEY = "264b34fc8f5b19c3d7242c85458cf41d5432b439c061a73e9436b0dc2b1c6563"
VIRUSTOTAL_URL_URL = "https://www.virustotal.com/api/v3/urls"

# -------------------------------
# Custom CSS for Modern Design
# -------------------------------
def add_custom_css():
    """
    Inject some custom CSS to give a more modern look to the app.
    """
    st.markdown(
        """
        <style>
        .main {
            background: linear-gradient(135deg, #fdfbfb 0%, #ebedee 100%);
        }
        h1, h2, h3, h4 {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            color: #2c3e50;
        }
        .stButton button {
            background-color: #2980b9 !important;
            color: #fff !important;
            border-radius: 6px !important;
        }
        [data-testid="stSidebar"] {
            background: #ecf0f1;
        }
        .css-1p05t2l table {
            border: 1px solid #ccc;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------
# Gemini Simulation Function
# -------------------------------
def analyze_email_with_gemini(email_content):
    """
    Simulated Gemini analysis.
    In a real implementation, you would call the Gemini API with a prompt like:
    "Do you think this email is phishing?"
    Here we use a simple heuristic based on suspicious keywords.
    """
    suspicious_keywords = ["urgent", "verify", "account", "password", "click", "update"]
    score = sum(1 for word in suspicious_keywords if word in email_content.lower())
    if score >= 2:
        return f"High phishing suspicion (score: {score}). The email appears to be phishing."
    else:
        return f"Low phishing suspicion (score: {score}). The email appears to be legitimate."

# -------------------------------
# Gmail Integration Function
# -------------------------------
def fetch_gmail_emails(username, password, limit=5):
    """
    Connect to Gmail via IMAP and fetch the last 'limit' emails from the inbox.
    Requirements:
      - IMAP must be enabled in your Gmail settings.
      - If using 2FA, you need an app-specific password.
    """
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select("inbox")
        result, data = mail.search(None, "ALL")
        email_ids = data[0].split()

        # Get only the last 'limit' email IDs
        email_ids = email_ids[-limit:]
        emails = []
        for eid in email_ids:
            result, msg_data = mail.fetch(eid, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = msg.get("subject", "No Subject")
            body = ""

            # If multipart, walk through the parts to find the plain text
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")

            # Optionally, you could parse attachments here:
            # for part in msg.walk():
            #     if part.get_content_maintype() == 'multipart':
            #         continue
            #     if part.get('Content-Disposition') is None:
            #         continue
            #     # If it's an attachment, you can process it or pass to a scanning function
            #     # ...
            emails.append({"subject": subject, "body": body})
        mail.logout()
        return emails
    except Exception as e:
        return {"error": str(e)}

# -------------------------------
# IDS Model Class
# -------------------------------
class IDSModel:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.trained_models = {}

        # Base models for stacking
        self.base_models = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
        ]
        # Final stacking ensemble
        self.stacking_model = StackingClassifier(
            estimators=self.base_models,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        )

    def preprocess_data(self, df):
        """
        Preprocess dataset:
          - Drop/replace NaN & infinite
          - Convert columns to numeric
          - Split train/test
          - Scale & PCA
          - SMOTE for imbalance
        """
        if "Label" not in df.columns:
            raise ValueError("Error: 'Label' column not found in dataset.")

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = self.label_encoder.fit_transform(df["Label"].astype(str))
        X = X.values

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Pipeline for scaling + PCA
        self.preprocessor = ImbPipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95))
        ])
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)

        # SMOTE for class imbalance
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        """
        Train the base models and the stacking ensemble.
        """
        for name, model in self.base_models:
            model.fit(X_train, y_train)
            self.trained_models[name] = model

        self.stacking_model.fit(X_train, y_train)
        return self.trained_models, self.stacking_model

    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all models (base + stacking) and return metrics.
        """
        results = {}
        # Evaluate each base model
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=1),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=1),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=1),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
        # Evaluate stacking
        y_pred_stack = self.stacking_model.predict(X_test)
        results["Stacking"] = {
            "accuracy": accuracy_score(y_test, y_pred_stack),
            "precision": precision_score(y_test, y_pred_stack, average="weighted", zero_division=1),
            "recall": recall_score(y_test, y_pred_stack, average="weighted", zero_division=1),
            "f1_score": f1_score(y_test, y_pred_stack, average="weighted", zero_division=1),
            "confusion_matrix": confusion_matrix(y_test, y_pred_stack).tolist()
        }
        return results

    async def async_scan(self, sample, delay=0.3):
        """
        Simulate scanning a single network packet asynchronously.
        """
        await asyncio.sleep(delay)
        pred = self.stacking_model.predict(sample.reshape(1, -1))[0]
        return pred

    async def simulate_real_time_detection(self, X_test, num_samples=10):
        """
        Simulate real-time detection on a subset of test data asynchronously.
        """
        predictions = []
        for i in range(min(num_samples, len(X_test))):
            sample = X_test[i]
            pred = await self.async_scan(sample)
            predictions.append(pred)
        return predictions

# -------------------------------
# Phishing Scanner Class (VirusTotal-based)
# -------------------------------
class PhishingScannerVT:
    """
    Handles scanning of URLs found in an email via VirusTotal.
    """
    def __init__(self, vt_api_key=YOUR_VIRUSTOTAL_API_KEY):
        self.vt_api_key = vt_api_key
        self.headers = {
            "Accept": "application/json",
            "x-apikey": self.vt_api_key
        }

    def extract_links(self, email_content):
        pattern = r"(https?://[^\s]+)"
        return re.findall(pattern, email_content, re.IGNORECASE)

    def check_url_virustotal(self, url):
        """
        Submits a URL to VirusTotal, returns the JSON result or an error.
        """
        try:
            data = {"url": url}
            response = requests.post(VIRUSTOTAL_URL_URL, headers=self.headers, data=data)
            if response.status_code == 200:
                analysis_id = response.json()["data"]["id"]
                analysis_url = f"{VIRUSTOTAL_URL_URL}/{analysis_id}"
                analysis_resp = requests.get(analysis_url, headers=self.headers)
                if analysis_resp.status_code == 200:
                    return analysis_resp.json()
                else:
                    return {"error": "Unable to retrieve analysis results."}
            else:
                return {"error": f"VirusTotal submission failed. Status: {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}

    def analyze_email(self, email_content):
        """
        Extracts URLs from email content, scans them on VirusTotal, and returns results.
        If no URLs found, returns a message.
        """
        urls = self.extract_links(email_content)
        if not urls:
            return {"message": "No URLs found in the email content."}
        results = {}
        for url in urls:
            vt_result = self.check_url_virustotal(url)
            results[url] = vt_result
        return results

# -------------------------------
# Cloud Security Module (Placeholder)
# -------------------------------
def scan_cloud_security():
    """
    Placeholder for scanning cloud configurations (AWS, Azure, GCP).
    Currently displays a sample table of findings.
    """
    st.markdown("### Cloud Security Posture Assessment")
    st.write("Enter your cloud environment details (e.g., AWS account ID, region, etc.)")
    cloud_provider = st.selectbox("Select Cloud Provider", ["AWS", "Azure", "GCP"])
    account_id = st.text_input("Account ID / Project ID")

    if st.button("Scan Cloud Configurations"):
        st.success(f"Scanned {cloud_provider} account '{account_id}'.")
        findings = pd.DataFrame({
            "Issue": ["Open S3 Bucket", "Insecure Security Group", "Publicly Accessible Database"],
            "Severity": ["High", "Medium", "High"],
            "Recommendation": [
                "Restrict bucket access",
                "Review inbound/outbound rules",
                "Enable firewall and encryption"
            ]
        })
        st.dataframe(findings)
        st.plotly_chart(px.bar(findings, x="Issue", y=["Severity"], title="Cloud Security Findings"))

# -------------------------------
# Compliance Module (Placeholder)
# -------------------------------
def check_compliance():
    """
    Placeholder for a compliance checker (ISO 27001, NIST, GDPR, etc.).
    """
    st.markdown("### Data & Compliance Checker")
    st.write("Answer a few questions to assess your organization's security compliance:")
    industry = st.selectbox("Industry", ["Finance", "Healthcare", "Retail", "Government", "Other"])
    company_size = st.selectbox("Company Size", ["Small", "Medium", "Large"])
    compliance_target = st.multiselect("Select Standards to Check", ["ISO 27001", "NIST", "GDPR", "UAE TRA", "PCI-DSS"])

    if st.button("Run Compliance Check"):
        st.success("Compliance scan completed.")
        recommendations = pd.DataFrame({
            "Standard": compliance_target if compliance_target else ["None Selected"],
            "Score": [85]*len(compliance_target) if compliance_target else [0],
            "Action": ["Update policies and encrypt sensitive data"]*len(compliance_target) if compliance_target else ["N/A"]
        })
        st.dataframe(recommendations)
        st.plotly_chart(px.bar(recommendations, x="Standard", y="Score", title="Compliance Score"))

# -------------------------------
# User Security Module (Placeholder)
# -------------------------------
def show_user_security_tips():
    """
    Provides personal cybersecurity tips and a short quiz for user awareness.
    """
    st.markdown("### Personal Cybersecurity & Education")
    st.write("Improve your digital hygiene with these tips:")

    tips = [
        "Use strong, unique passwords and consider a password manager.",
        "Enable two-factor authentication (2FA) wherever possible.",
        "Be cautious of unsolicited emails and phishing links.",
        "Keep your software and operating systems up to date.",
        "Regularly backup your data."
    ]
    for tip in tips:
        st.write(f"• {tip}")

    st.markdown("#### Quick Quiz")
    st.write("Test your knowledge: What should you do if you suspect a phishing email?")
    options = st.radio("Choose the best answer:", [
        "Click on the link to see where it leads.",
        "Report it to your IT department and delete it.",
        "Reply asking for more details.",
        "Forward it to your colleagues."
    ])
    if st.button("Submit Answer"):
        if options == "Report it to your IT department and delete it.":
            st.success("Correct! Reporting and deleting suspicious emails is the best practice.")
        else:
            st.error("Incorrect. The best practice is to report and delete suspicious emails immediately.")

# -------------------------------
# Unified Dashboard
# -------------------------------
def show_dashboard():
    """
    A high-level dashboard that aggregates metrics from different modules.
    """
    st.markdown("### Unified Security Dashboard")
    st.write("Welcome to the all-in-one cybersecurity suite. Here is an overview of your environment:")
    summary_data = {
        "IDS Alerts": 5,
        "Phishing Scans": 3,
        "Cloud Findings": 2,
        "Compliance Score": 88,
        "User Awareness": "Yes"
    }
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("IDS Alerts", summary_data["IDS Alerts"])
    col2.metric("Phishing Scans", summary_data["Phishing Scans"])
    col3.metric("Cloud Findings", summary_data["Cloud Findings"])
    col4.metric("Compliance Score", summary_data["Compliance Score"])
    col5.metric("User Awareness", summary_data["User Awareness"])

    st.plotly_chart(
        px.pie(
            names=list(summary_data.keys()),
            values=list(summary_data.values()),
            title="Overall Security Posture"
        )
    )

# -------------------------------
# Phishing Scanner Module with Enhanced Modes
# -------------------------------
def phishing_scanner_module():
    """
    Allows direct analysis (text/image) or Gmail inbox scanning for phishing detection.
    """
    st.subheader("✉️ Phishing Email Scanner")
    scanner_mode = st.radio("Select Mode", ["Direct Analysis", "Gmail Inbox"])

    if scanner_mode == "Direct Analysis":
        st.write("Provide email text and/or upload an image for analysis.")
        email_content = st.text_area("Email Content", height=150)
        screenshot_file = st.file_uploader("Upload Screenshot (PNG/JPG) (Optional)", type=["png", "jpg", "jpeg"])

        if st.button("Analyze Email"):
            result_message = ""
            # 1. If user typed text
            if email_content.strip():
                # Check for URLs
                urls = PhishingScannerVT().extract_links(email_content)
                if urls:
                    # Use VirusTotal if URLs are detected
                    scanner = PhishingScannerVT()
                    vt_results = scanner.analyze_email(email_content)
                    result_message = "VirusTotal Analysis:\n" + str(vt_results)
                else:
                    # If no URLs, fallback to Gemini simulation
                    result_message = "Gemini Analysis:\n" + analyze_email_with_gemini(email_content)

            # 2. If no text but an image is uploaded
            elif screenshot_file is not None:
                try:
                    image = Image.open(screenshot_file)
                    extracted_text = pytesseract.image_to_string(image)
                    if extracted_text.strip():
                        result_message = "Gemini Analysis (from image):\n" + analyze_email_with_gemini(extracted_text)
                    else:
                        result_message = "No text could be extracted from the image."
                except Exception as e:
                    result_message = f"Error processing image: {e}"

            else:
                result_message = "Please provide email content or upload an image."

            st.write(result_message)

    elif scanner_mode == "Gmail Inbox":
        st.write("Log in to your Gmail account to fetch your latest emails.")
        gmail_user = st.text_input("Gmail Address")
        gmail_pass = st.text_input("Gmail Password", type="password")

        if st.button("Fetch Emails"):
            emails = fetch_gmail_emails(gmail_user, gmail_pass)
            if isinstance(emails, dict) and "error" in emails:
                st.error("Error fetching emails: " + emails["error"])
            else:
                st.session_state.emails = emails
                st.success("Fetched last 5 emails successfully.")

        if "emails" in st.session_state:
            email_choice = st.selectbox("Select an Email", st.session_state.emails, format_func=lambda x: x["subject"])

            if st.button("Analyze Selected Email"):
                # Display the email subject and body
                st.write("### Selected Email Content")
                st.write(f"**Subject:** {email_choice['subject']}")
                st.write(f"**Body:** {email_choice['body']}")

                # Now run Gemini analysis on the email body
                email_text = email_choice["body"]
                result_message = "Gemini Analysis:\n" + analyze_email_with_gemini(email_text)

                st.write("### Gemini Analysis Result:")
                st.write(result_message)

# -------------------------------
# Main Function
# -------------------------------
def main():
    add_custom_css()
    st.title("🔐 Unified Cybersecurity Suite")
    st.markdown("""
    <h4 style='color:#7f8c8d;'>An all-in-one solution covering Threat Detection, Data Compliance, Cloud Defense, and Personal Cybersecurity.</h4>
    """, unsafe_allow_html=True)

    st.sidebar.header("Navigation")
    menu_options = [
        "Dashboard",
        "Intrusion Detection (IDS)",
        "Phishing Scanner",
        "Cloud Security",
        "Compliance",
        "User Security"
    ]
    choice = st.sidebar.selectbox("Select Module", menu_options)

    # Keep IDS model in session state so it doesn't retrain each time
    if "ids_model" not in st.session_state:
        st.session_state.ids_model = IDSModel()

    if choice == "Dashboard":
        show_dashboard()

    elif choice == "Intrusion Detection (IDS)":
        st.subheader("🚀 Intrusion Detection System (IDS)")
        uploaded_file = st.file_uploader("Upload Network Traffic CSV", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("### Data Preview")
            st.dataframe(df.head())

            try:
                X_train, X_test, y_train, y_test = st.session_state.ids_model.preprocess_data(df)
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.df_uploaded = True
                st.success("✅ Data preprocessed successfully!")
            except Exception as e:
                st.error(f"Preprocessing Error: {e}")
                st.stop()

            if st.button("Train Models"):
                with st.spinner("Training IDS models..."):
                    st.session_state.ids_model.train_models(X_train, y_train)
                st.success("✅ Models trained successfully!")

                # Evaluate
                results = st.session_state.ids_model.evaluate_models(X_test, y_test)
                st.subheader("📊 Model Performance")
                for model_name, metrics in results.items():
                    st.write(f"#### {model_name} Results")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", round(metrics["accuracy"], 4))
                    col2.metric("Precision", round(metrics["precision"], 4))
                    col3.metric("Recall", round(metrics["recall"], 4))
                    col4.metric("F1-Score", round(metrics["f1_score"], 4))

                    fig, ax = plt.subplots()
                    sns.heatmap(np.array(metrics["confusion_matrix"]), annot=True, fmt="d", cmap="Blues", ax=ax)
                    ax.set_title(f"{model_name} Confusion Matrix")
                    st.pyplot(fig)

                st.markdown("### Before vs. After Using Our IDS")
                st.write("""
                **Before**: Intrusions may go undetected or yield false positives.
                **After**: Our ensemble model improves detection accuracy with real-time monitoring.
                """)

        if st.sidebar.button("⏱ Real-Time Threat Simulation"):
            # Simulate real-time detection if models are trained
            if not st.session_state.get("df_uploaded", False):
                st.error("Please upload and train on a dataset first.")
            else:
                if not st.session_state.ids_model.trained_models:
                    st.error("Models have not been trained yet! Please train them.")
                else:
                    st.subheader("⏱ Real-Time Threat Detection Simulation")

                    async def run_simulation():
                        return await st.session_state.ids_model.simulate_real_time_detection(
                            st.session_state.X_test, num_samples=10
                        )

                    predictions = asyncio.run(run_simulation())
                    st.write("### Detected Threats:")
                    for idx, pred in enumerate(predictions):
                        label = st.session_state.ids_model.label_encoder.inverse_transform([int(pred)])[0]
                        st.write(f"Packet {idx + 1}: **{label}**")

                    df_preds = pd.DataFrame({"Prediction": predictions})
                    fig = px.bar(df_preds, x=df_preds.index, y="Prediction", title="Real-Time Threats")
                    st.plotly_chart(fig)

    elif choice == "Phishing Scanner":
        phishing_scanner_module()

    elif choice == "Cloud Security":
        scan_cloud_security()

    elif choice == "Compliance":
        check_compliance()

    elif choice == "User Security":
        show_user_security_tips()



if __name__ == "__main__":
    main()
