import streamlit as st

st.set_page_config(page_title="Unified Cybersecurity Suite", layout="wide")
import plotly.express as px
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import asyncio
import requests
import re
import imaplib
import email
from concurrent.futures import ThreadPoolExecutor
import random
from datetime import datetime, timedelta
import joblib
import warnings
import urllib3
import boto3
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
# Suppress warnings from XGBoost and insecure requests
warnings.filterwarnings("ignore", message="Parameters: { 'use_label_encoder' } are not used.")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

########################################
# Set Tesseract path if needed:
if not os.path.isfile(pytesseract.pytesseract.tesseract_cmd):
    default_tesseract = "/usr/bin/tesseract"  # Change this if you're on Windows
    if os.path.isfile(default_tesseract):
        pytesseract.pytesseract.tesseract_cmd = default_tesseract
    else:
        st.error(
            "Tesseract OCR executable not found. Please install Tesseract and set pytesseract.pytesseract.tesseract_cmd appropriately.")
########################################

# -------------------------------
# CONFIGURATION & CONSTANTS
# -------------------------------
YOUR_VIRUSTOTAL_API_KEY = "264b34fc8f5b19c3d7242c85458cf41d5432b439c061a73e9436b0dc2b1c6563"
VIRUSTOTAL_URL_URL = "https://www.virustotal.com/api/v3/urls"


# -------------------------------
# Custom CSS for Modern Design
# -------------------------------
def add_custom_css():
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
# Enhanced Gemini Simulation (Phishing Risk Calculator)
# -------------------------------
def analyze_email_with_gemini(email_content):
    suspicious_keywords = [
        "urgent", "verify", "account", "password", "click", "update", "login",
        "confirm", "bank", "credit", "security", "breach", "alert", "wire",
        "refund", "limited", "locked", "suspended", "deactivate", "suntrust"
    ]

    total_keywords = len(suspicious_keywords)
    count = sum(1 for word in suspicious_keywords if word in email_content.lower())
    risk_percentage = (count / total_keywords) * 100
    if risk_percentage < 40:
        risk_level = "Level 1 (Low risk)"
    elif risk_percentage < 70:
        risk_level = "Level 2 (Moderate risk)"
    else:
        risk_level = "Level 3 (High risk)"
    return f"Phishing Risk: {risk_percentage:.0f}% - {risk_level}"


# -------------------------------
# Gmail Integration Function
# -------------------------------
def fetch_gmail_emails(username, password, limit=5):
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select("inbox")
        result, data = mail.search(None, "ALL")
        email_ids = data[0].split()
        email_ids = email_ids[-limit:]
        emails = []
        for eid in email_ids:
            result, msg_data = mail.fetch(eid, "(RFC822)")
            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)
            subject = msg.get("subject", "No Subject")
            body = ""
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    content_disposition = str(part.get("Content-Disposition"))
                    if content_type == "text/plain" and "attachment" not in content_disposition:
                        body = part.get_payload(decode=True).decode(errors="ignore")
                        break
            else:
                body = msg.get_payload(decode=True).decode(errors="ignore")
            emails.append({"subject": subject, "body": body})
        mail.logout()
        return emails
    except Exception as e:
        return {"error": str(e)}


# -------------------------------
# IDS Model Class with Persistence
# -------------------------------
class IDSModel:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.preprocessor = None
        self.trained_models = {}
        self.base_models = [
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("xgb", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42)),
        ]
        self.stacking_model = StackingClassifier(
            estimators=self.base_models,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
        )

    def preprocess_data(self, df):
        if "Label" not in df.columns:
            raise ValueError("Error: 'Label' column not found in dataset.")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        X = df.drop(columns=["Label"]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = self.label_encoder.fit_transform(df["Label"].astype(str))
        X = X.values
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        self.preprocessor = ImbPipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=0.95))
        ])
        X_train = self.preprocessor.fit_transform(X_train)
        X_test = self.preprocessor.transform(X_test)
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        return X_train, X_test, y_train, y_test

    def train_models(self, X_train, y_train):
        model_file = "trained_ids_model.pkl"
        if os.path.exists(model_file):
            self.trained_models, self.stacking_model = joblib.load(model_file)
            st.info("Loaded trained IDS model from disk.")
        else:
            for name, model in self.base_models:
                model.fit(X_train, y_train)
                self.trained_models[name] = model
            base_predictions = np.column_stack([model.predict(X_train) for model in self.trained_models.values()])
            self.stacking_model.fit(base_predictions, y_train)
            joblib.dump((self.trained_models, self.stacking_model), model_file)
            st.success("Trained IDS model saved to disk.")
        return self.trained_models, self.stacking_model

    def evaluate_models(self, X_test, y_test):
        results = {}
        for name, model in self.trained_models.items():
            y_pred = model.predict(X_test)
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=1),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=1),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=1),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
            }
        base_predictions = np.column_stack([model.predict(X_test) for model in self.trained_models.values()])
        y_pred_stack = self.stacking_model.predict(base_predictions)
        results["Stacking"] = {
            "accuracy": accuracy_score(y_test, y_pred_stack),
            "precision": precision_score(y_test, y_pred_stack, average="weighted", zero_division=1),
            "recall": recall_score(y_test, y_pred_stack, average="weighted", zero_division=1),
            "f1_score": f1_score(y_test, y_pred_stack, average="weighted", zero_division=1),
            "confusion_matrix": confusion_matrix(y_test, y_pred_stack).tolist()
        }
        return results

    async def async_scan(self, sample, delay=0.3):
        await asyncio.sleep(delay)
        # Instead of passing raw features, get predictions from each base model first.
        base_preds = np.column_stack([model.predict(sample.reshape(1, -1)) for model in self.trained_models.values()])
        pred = self.stacking_model.predict(base_preds)[0]
        return pred

    async def simulate_real_time_detection(self, X_test, num_samples=10):
        predictions = []
        for i in range(min(num_samples, len(X_test))):
            sample = X_test[i]
            pred = await self.async_scan(sample)
            predictions.append(pred)
        return predictions



###############################
# 1) Enhanced Gemini Analysis #
###############################

def analyze_email_with_gemini(email_content):
    """
    Enhanced 'Gemini' heuristic for phishing risk.
    Now includes a larger set of suspicious keywords to produce more varied scores.
    """
    # Expanded suspicious keyword list
    suspicious_keywords = [
        "urgent", "verify", "account", "password", "click", "update", "free", "login", "confirm",
        "credit card", "bank", "alert", "security", "breach", "locked", "suspended", "limited",
        "scam", "request", "wire", "transfer", "refund"
    ]
    total_keywords = len(suspicious_keywords)
    count = 0

    # Convert to lowercase once for performance
    lower_content = email_content.lower()

    # Count how many suspicious keywords appear
    for kw in suspicious_keywords:
        if kw in lower_content:
            count += 1

    # Risk percentage
    risk_percentage = (count / total_keywords) * 100

    if risk_percentage < 40:
        risk_level = "Level 1 (Low risk)"
    elif risk_percentage < 70:
        risk_level = "Level 2 (Moderate risk)"
    else:
        risk_level = "Level 3 (High risk)"

    return f"Phishing Risk: {risk_percentage:.0f}% - {risk_level}"


#####################################
# 2) Enhanced VirusTotal Scanner    #
#####################################

class PhishingScannerVT:
    def __init__(self, vt_api_key=None):
        if vt_api_key is None:
            vt_api_key = YOUR_VIRUSTOTAL_API_KEY
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
        Submits a URL to VirusTotal, returns JSON or an error dict.
        Includes a fallback note if there's an error.
        """
        try:
            data = {"url": url}
            # Disable SSL verify if needed (for testing)
            response = requests.post(
                "https://www.virustotal.com/api/v3/urls",
                headers=self.headers,
                data=data,
                verify=False  # Not recommended in production
            )
            if response.status_code == 200:
                analysis_id = response.json()["data"]["id"]
                analysis_url = f"https://www.virustotal.com/api/v3/urls/{analysis_id}"
                analysis_resp = requests.get(analysis_url, headers=self.headers, verify=False)
                if analysis_resp.status_code == 200:
                    return analysis_resp.json()
                else:
                    return {
                        "error": "Unable to retrieve analysis results.",
                        "fallback_note": "Link may still be suspicious."
                    }
            else:
                return {
                    "error": f"VirusTotal submission failed. Status: {response.status_code}",
                    "fallback_note": "Link may still be suspicious."
                }
        except Exception as e:
            return {
                "error": str(e),
                "fallback_note": "Link may still be suspicious."
            }

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
# Threat Intelligence Feed Module (Enhanced with CSV Upload)
# -------------------------------
def show_threat_intel_feed():
    st.markdown("### Threat Intelligence Feed")
    st.write("Upload a CSV file containing threat indicators.")
    uploaded_file = st.file_uploader("Upload Threat Intelligence CSV", type=["csv"])
    if uploaded_file:
        df_intel = pd.read_csv(uploaded_file)
        # Rename columns if needed
        if "Indicator" not in df_intel.columns and "indicator" in df_intel.columns:
            df_intel = df_intel.rename(columns={"indicator": "Indicator"})
        if "Severity" not in df_intel.columns and "classification" in df_intel.columns:
            df_intel = df_intel.rename(columns={"classification": "Severity"})
        if "Timestamp" not in df_intel.columns and "detected_date" in df_intel.columns:
            df_intel = df_intel.rename(columns={"detected_date": "Timestamp"})
        st.write("### Threat Intelligence Data")
        st.dataframe(df_intel)
        if "Severity" in df_intel.columns:
            severity_options = df_intel["Severity"].unique().tolist()
            severity_filter = st.multiselect("Filter by Severity", options=severity_options, default=severity_options)
            df_filtered = df_intel[df_intel["Severity"].isin(severity_filter)]
        else:
            df_filtered = df_intel
        st.dataframe(df_filtered)
        st.plotly_chart(px.bar(df_filtered, x="Indicator", y=df_filtered.index, color="Severity",
                               title="Threat Indicators (Filtered)"))
        st.session_state["threat_intel"] = len(df_filtered)
    else:
        st.info("Please upload a CSV file with your threat intelligence data.")


# -------------------------------
# Cloud Security Module (Enhanced with CSV Upload)
# -------------------------------
# AWS Configuration (Replace with actual credentials or use AWS Profile)
AWS_ACCESS_KEY = "AKIAU6GDVM66RZT37LOU"  # ❌ Replace or use AWS Profile
AWS_SECRET_KEY = "8vsjj5QabRyY5VEfnNokJ8g9HR7BjjovQO3Qrmt7"  # ❌ Replace or use AWS Profile
AWS_REGION = "us-east-1"
LOG_GROUP_NAME = "Honeypot-SSH-Logs"
LOG_STREAM_NAME = "i-00f2acb92268f3a39"  # ✅ Correct log stream name

# Initialize Boto3 Client
client = boto3.client(
    "logs",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

def scan_cloud_security():
    """Fetch CloudWatch logs and display in Streamlit."""
    st.markdown("### 🔒 Cloud Security Posture")
    st.write("Fetching honeypot logs from AWS CloudWatch...")

    if st.button("Fetch Logs"):
        try:
            response = client.get_log_events(
                logGroupName=LOG_GROUP_NAME,
                logStreamName=LOG_STREAM_NAME,
                limit=10
            )

            events = response.get("events", [])
            if events:
                # Convert log events to DataFrame
                df_logs = pd.DataFrame(events)
                if "timestamp" in df_logs.columns:
                    df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"], unit="ms")
                st.write("### 📜 Honeypot Logs")
                st.dataframe(df_logs)
                st.session_state["cloud_findings"] = len(df_logs)
            else:
                st.info("No logs found for the specified log group/stream.")
        except Exception as e:
            st.error(f"❌ Error fetching logs: {str(e)}")





# -------------------------------
# AI Security Quiz Module (Replaces Compliance)
# -------------------------------
def ai_security_quiz():
    # Define quiz questions
    questions = [
        {
            "question": "What is the most secure way to manage your passwords?",
            "choices": [
                "Write them on a sticky note",
                "Reuse the same password for every account",
                "Use a password manager",
                "Share them with a trusted friend"
            ],
            "answer": "Use a password manager"
        },
        {
            "question": "Which of the following is a strong indicator of a phishing email?",
            "choices": [
                "An email from your bank asking to update your password",
                "An email from a known friend",
                "An official company newsletter",
                "An automated bill reminder"
            ],
            "answer": "An email from your bank asking to update your password"
        },
        {
            "question": "What should you do if you receive an unsolicited email with a suspicious attachment?",
            "choices": [
                "Open the attachment to see what's inside",
                "Forward it to colleagues",
                "Delete the email and report it",
                "Reply to ask for more information"
            ],
            "answer": "Delete the email and report it"
        },
        {
            "question": "Which best describes two-factor authentication (2FA)?",
            "choices": [
                "Using two different passwords",
                "A process requiring two forms of identification",
                "A single password with a security question",
                "Encrypting data twice"
            ],
            "answer": "A process requiring two forms of identification"
        }
    ]

    # Initialize session state for quiz if not already set
    if "quiz_question" not in st.session_state:
        st.session_state.quiz_question = random.choice(questions)
        st.session_state.quiz_submitted = False
        st.session_state.quiz_feedback = ""

    quiz_container = st.container()

    # Display quiz inside the container
    if not st.session_state.quiz_submitted:
        quiz_container.write("**Question:** " + st.session_state.quiz_question["question"])
        selected = quiz_container.radio("Choose an answer:", st.session_state.quiz_question["choices"], key="quiz_radio")
        if quiz_container.button("Submit Answer"):
            if selected == st.session_state.quiz_question["answer"]:
                st.session_state.quiz_feedback = "Correct!"
                st.session_state["compliance_score"] = st.session_state.get("compliance_score", 0) + 10
                st.session_state.user_awareness = "Yes"
            else:
                st.session_state.quiz_feedback = "Incorrect. The correct answer is: " + st.session_state.quiz_question["answer"]
                st.session_state.user_awareness = "No"
            st.session_state.quiz_submitted = True
            quiz_container.empty()  # Clear container to force re-render
            ai_security_quiz()  # Re-call the function to update UI
    else:
        quiz_container.write("**Feedback:** " + st.session_state.quiz_feedback)
        if quiz_container.button("Next Question"):
            st.session_state.quiz_question = random.choice(questions)
            st.session_state.quiz_submitted = False
            st.session_state.quiz_feedback = ""
            quiz_container.empty()
            ai_security_quiz()

# Usage: call ai_security_quiz() in your main app when you want to show the quiz.
# -------------------------------
# Incident Response & Remediation Module (New Feature)
# -------------------------------
def incident_response():
    st.markdown("### Incident Response & Remediation")
    # Gather current security metrics from session state
    ids_alerts = st.session_state.get("ids_alerts", 0)
    phishing_scans = st.session_state.get("phishing_scans", 0)
    cloud_findings = st.session_state.get("cloud_findings", 0)
    threat_intel = st.session_state.get("threat_intel", 0)
    user_awareness = st.session_state.get("user_awareness", "No")

    # Construct an AI prompt based on current metrics
    prompt = (
        "Analyze the following security metrics and generate an incident response plan:\n"
        f"- IDS Alerts: {ids_alerts}\n"
        f"- Phishing Scans: {phishing_scans}\n"
        f"- Cloud Security Findings: {cloud_findings}\n"
        f"- Threat Intelligence Indicators: {threat_intel}\n"
        f"- User Awareness: {user_awareness}\n\n"
        "Based on these metrics, recommend the best course of action to remediate potential incidents, "
        "minimize risks, and improve overall security posture."
    )



    # Simulated analysis: generate recommendations based on thresholds
    recommendations = []
    if ids_alerts > 10:
        recommendations.append("Immediately investigate network traffic anomalies and isolate affected segments.")
    elif ids_alerts > 5:
        recommendations.append("Review IDS alerts and verify unusual traffic patterns.")

    if phishing_scans > 5:
        recommendations.append("Conduct a comprehensive phishing investigation and enforce immediate password resets for impacted users.")
    elif phishing_scans > 3:
        recommendations.append("Review recent phishing alerts and reinforce email filtering rules.")

    if cloud_findings > 2:
        recommendations.append("Assess and remediate cloud configuration vulnerabilities promptly.")

    if threat_intel > 0:
        recommendations.append("Correlate threat intelligence data with internal logs to identify and neutralize potential threats.")

    if user_awareness == "No":
        recommendations.append("Schedule a mandatory security awareness training for all users.")

    if not recommendations:
        recommendations.append("Current metrics indicate low risk; continue with routine monitoring and regular security reviews.")

    # Combine recommendations into a final response plan
    response_plan = "Based on the analysis, the following incident response plan is recommended:\n" + "\n".join(f"- {rec}" for rec in recommendations)

    st.write("#### AI-Generated Incident Response Plan:")
    st.write("Response Plan", response_plan, height=400)


# -------------------------------
# Dynamic Dashboard
# -------------------------------
def show_dashboard():
    st.markdown("### Dynamic Security Dashboard")
    st.write("Welcome to the all-in-one cybersecurity suite. Here is an overview of your environment:")
    ids_alerts = st.session_state.get("ids_alerts", 0)
    phishing_scans = st.session_state.get("phishing_scans", 0)
    cloud_findings = st.session_state.get("cloud_findings", 0)
    threat_intel = st.session_state.get("threat_intel", 0)
    user_awareness = st.session_state.get("user_awareness", "No")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("IDS Alerts", ids_alerts)
    col2.metric("Phishing Scans", phishing_scans)
    col3.metric("Cloud Findings", cloud_findings)
    col4.metric("Threat Intel", threat_intel)
    col5.metric("User Awareness", user_awareness)
    numeric_data = [
        ids_alerts,
        phishing_scans,
        cloud_findings,
        threat_intel,
        1 if user_awareness == "Yes" else 0
    ]
    categories = ["IDS Alerts", "Phishing Scans", "Cloud Findings", "Threat Intel", "User Awareness"]
    st.plotly_chart(
        px.pie(
            names=categories,
            values=numeric_data,
            title="Overall Security Posture"
        )
    )


# -------------------------------
# Phishing Scanner Module with Enhanced Modes
# -------------------------------
def phishing_scanner_module():
    st.subheader("✉️ Phishing Email Scanner")
    scanner_mode = st.radio("Select Mode", ["Direct Analysis", "Gmail Inbox"])
    if scanner_mode == "Direct Analysis":
        st.write("Provide email text and/or upload an image for analysis.")
        email_content = st.text_area("Email Content", height=150)
        screenshot_file = st.file_uploader("Upload Screenshot (PNG/JPG) (Optional)", type=["png", "jpg", "jpeg"])
        if st.button("Analyze Email"):
            st.session_state["phishing_scans"] = st.session_state.get("phishing_scans", 0) + 1
            result_message = ""
            if email_content.strip():
                urls = PhishingScannerVT().extract_links(email_content)
                if urls:
                    scanner = PhishingScannerVT()
                    vt_results = scanner.analyze_email(email_content)
                    result_message = "VirusTotal Analysis:\n" + str(vt_results)
                else:
                    result_message = "Gemini Analysis:\n" + analyze_email_with_gemini(email_content)
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
                st.session_state["phishing_scans"] = st.session_state.get("phishing_scans", 0) + 1
                st.write("### Selected Email Content")
                st.write(f"**Subject:** {email_choice['subject']}")
                st.write(f"**Body:** {email_choice['body']}")
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
    <h4 style='color:#7f8c8d;'>An all-in-one solution covering Threat Detection, AI-Driven Security Quizzes, Cloud Defense, Threat Intelligence, and Incident Response.</h4>
    """, unsafe_allow_html=True)
    st.sidebar.header("Navigation")
    menu_options = ["Dashboard", "Intrusion Detection (IDS)", "Phishing Scanner", "Cloud Security",
                    "Threat Intelligence Feed", "AI Security Quiz", "Incident Response"]
    choice = st.sidebar.selectbox("Select Module", menu_options)

    # Initialize session state variables if not present
    if "ids_model" not in st.session_state:
        st.session_state.ids_model = IDSModel()
    if "ids_alerts" not in st.session_state:
        st.session_state["ids_alerts"] = 0
    if "phishing_scans" not in st.session_state:
        st.session_state["phishing_scans"] = 0
    if "cloud_findings" not in st.session_state:
        st.session_state["cloud_findings"] = 0
    if "threat_intel" not in st.session_state:
        st.session_state["threat_intel"] = 0
    if "user_awareness" not in st.session_state:
        st.session_state["user_awareness"] = "No"
    if "compliance_score" in st.session_state:
        del st.session_state["compliance_score"]

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
                if not st.session_state.get("trained_ids_model", False):
                    with st.spinner("Training IDS models..."):
                        st.session_state.ids_model.train_models(X_train, y_train)
                    st.session_state["trained_ids_model"] = True
                    st.success("✅ Models trained successfully!")
                else:
                    st.info("Models already trained. To retrain, please clear the session state.")
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
            if not st.session_state.get("df_uploaded", False):
                st.error("Please upload and train on a dataset first.")
            else:
                if not st.session_state.ids_model.trained_models:
                    st.error("Models have not been trained yet! Please train them.")
                else:
                    st.subheader("⏱ Real-Time Threat Detection Simulation")

                    async def run_simulation():
                        return await st.session_state.ids_model.simulate_real_time_detection(st.session_state.X_test,
                                                                                             num_samples=10)

                    predictions = asyncio.run(run_simulation())
                    st.write("### Detected Threats:")
                    intrusion_count = 0
                    for idx, pred in enumerate(predictions):
                        label = st.session_state.ids_model.label_encoder.inverse_transform([int(pred)])[0]
                        st.write(f"Packet {idx + 1}: **{label}**")
                        if label.lower() != "normal":
                            intrusion_count += 1
                    st.session_state["ids_alerts"] = st.session_state.get("ids_alerts", 0) + intrusion_count
                    df_preds = pd.DataFrame({"Prediction": predictions})
                    fig = px.bar(df_preds, x=df_preds.index, y="Prediction", title="Real-Time Threats")
                    st.plotly_chart(fig)
    elif choice == "Phishing Scanner":
        phishing_scanner_module()
    elif choice == "Cloud Security":
        scan_cloud_security()
    elif choice == "Threat Intelligence Feed":
        show_threat_intel_feed()
    elif choice == "AI Security Quiz":
        ai_security_quiz()
    elif choice == "Incident Response":
        incident_response()


if __name__ == "__main__":
    main()
