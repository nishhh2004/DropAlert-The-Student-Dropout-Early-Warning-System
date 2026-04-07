import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ---------------- WEB APP CONFIG ----------------
st.set_page_config(page_title="DropAlert App", page_icon="⚡", layout="wide")

@st.cache_resource
def load_model():
    try:
        return pickle.load(open("model.pkl", "rb"))
    except FileNotFoundError:
        st.error("Model file not found. Please ensure 'model.pkl' is in the same directory.")
        return None

model = load_model()

# ---------------- MODERN SAAS CSS ----------------
st.markdown("""
<style>
/* Base Web App Colors - Deep Space Dark Theme */
.stApp {
    background-color: #0B0F19; /* Very dark blue/black */
    color: #E2E8F0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hide Streamlit Default UI Elements EXCEPT the header so menus still work if needed */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Typography */
.brand-header {
    font-size: 28px;
    font-weight: 900;
    color: #FFFFFF;
    margin-bottom: 20px;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 15px;
}
.app-header {
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 5px;
    letter-spacing: -0.5px;
}
.app-subtitle {
    font-size: 15px;
    color: #94A3B8;
    margin-bottom: 30px;
}

/* Web App Cards */
.form-card {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 16px;
    padding: 30px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}

.result-card {
    background: linear-gradient(145deg, #1A233A 0%, #111827 100%);
    border: 1px solid #2D3748;
    border-radius: 16px;
    padding: 35px 30px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.4);
    text-align: center;
}

/* Form Section Headers */
.section-title {
    font-size: 14px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #60A5FA;
    font-weight: 600;
    margin-bottom: 15px;
    border-bottom: 1px solid #1F2937;
    padding-bottom: 8px;
}

/* Status Badges */
.status-badge {
    display: inline-block;
    padding: 8px 24px;
    border-radius: 9999px;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 1px;
    margin-bottom: 20px;
}
.bg-safe { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
.bg-risk { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
.bg-critical { background-color: rgba(239, 68, 68, 0.15); color: #F87171; border: 1px solid rgba(239, 68, 68, 0.3); }

/* Custom Action Button */
div.stButton > button:first-child {
    width: 100%;
    background: linear-gradient(90deg, #3B82F6 0%, #8B5CF6 100%);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 0px;
    font-weight: 600;
    font-size: 16px;
    margin-top: 20px;
    transition: all 0.3s ease;
}
div.stButton > button:first-child:hover {
    box-shadow: 0 0 20px rgba(139, 92, 246, 0.4);
    transform: translateY(-2px);
    color: white;
}

/* Alert Boxes for Recommendations */
.alert-box {
    text-align: left;
    background-color: #1F2937;
    border-left: 4px solid #3B82F6;
    padding: 12px 16px;
    border-radius: 6px;
    margin-top: 15px;
    font-size: 14px;
    color: #E2E8F0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- BRANDING HEADER ----------------
st.markdown("<div class='brand-header'>⚡ DropAlert<span style='color: #3B82F6;'></span></div>", unsafe_allow_html=True)

# ---------------- MAIN APP AREA ----------------
st.markdown('<div class="app-header">Student Risk Assessment</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Enter student metrics below to run the predictive dropout analysis model.</div>', unsafe_allow_html=True)

# Split screen: Form on left, Results on right
col1, col2 = st.columns([1.2, 1], gap="large")

# ---------------- LEFT: DATA ENTRY FORM ----------------
with col1:
    st.markdown('<div class="form-card">', unsafe_allow_html=True)
    
    # Section 1
    st.markdown('<div class="section-title">Academic Metrics</div>', unsafe_allow_html=True)
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        attendance = st.number_input("Attendance (%)", min_value=0, max_value=100, value=60)
        assignments = st.number_input("Assignments Submitted", min_value=0, max_value=50, value=10)
    with f_col2:
        cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=10.0, value=6.0, step=0.1)
        lab_hours = st.number_input("Weekly Lab Hours", min_value=0, max_value=40, value=5)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Section 2
    st.markdown('<div class="section-title">External Factors</div>', unsafe_allow_html=True)
    f_col3, f_col4 = st.columns(2)
    with f_col3:
        part_time = st.selectbox("Has Part-Time Job?", ["No", "Yes"])
        scholarship = st.selectbox("Receives Scholarship?", ["No", "Yes"])
    with f_col4:
        counsellor = st.number_input("Counsellor Visits", min_value=0, max_value=20, value=1)
        
    submit = st.button("Generate Risk Profile")
    st.markdown('</div>', unsafe_allow_html=True)

# Convert strings for the model
part_time_val = 1 if part_time == "Yes" else 0
scholarship_val = 1 if scholarship == "Yes" else 0

# ---------------- RIGHT: RESULT ENGINE ----------------
with col2:
    if submit and model is not None:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        
        # 1. Run Prediction
        X_input = pd.DataFrame([[attendance, cgpa, assignments, lab_hours, part_time_val, scholarship_val, counsellor]], 
                               columns=['Attendance_%', 'CGPA', 'Assignments_Submitted', 'Lab_Hours_Per_Week', 
                                        'Has_Part_Time_Job', 'Scholarship', 'Counsellor_Visits'])
        prob = model.predict_proba(X_input)[0][1]
        
        # 2. Status Logic
        if prob <= 0.3:
            status, cls, color = "SAFE", "bg-safe", "#34D399"
        elif prob <= 0.6:
            status, cls, color = "AT RISK", "bg-risk", "#FBBF24"
        else:
            status, cls, color = "CRITICAL", "bg-critical", "#F87171"

        # 3. UI Display
        st.markdown("<div style='color: #94A3B8; font-size: 14px; margin-bottom: 10px; text-transform: uppercase;'>Analysis Complete</div>", unsafe_allow_html=True)
        st.markdown(f'<div class="status-badge {cls}">STATUS: {status}</div>', unsafe_allow_html=True)
        
        st.markdown(f"<div style='font-size: 48px; font-weight: 800; color: {color}; line-height: 1;'>{prob*100:.1f}%</div>", unsafe_allow_html=True)
        st.markdown("<div style='color: #64748B; font-size: 13px; margin-bottom: 25px;'>Probability of Dropout</div>", unsafe_allow_html=True)
        
        # 4. Action Engine
        st.markdown("<hr style='border-color: #2D3748;'>", unsafe_allow_html=True)
        st.markdown("<div style='text-align: left; color: #E2E8F0; font-weight: 600; margin-bottom: 10px;'>System Action Items:</div>", unsafe_allow_html=True)
        
        if prob <= 0.3:
             st.markdown('<div class="alert-box" style="border-left-color: #34D399;">✓ Student is currently stable. No immediate administrative action required.</div>', unsafe_allow_html=True)
        else:
            if attendance < 60:
                st.markdown('<div class="alert-box" style="border-left-color: #F87171;">⚠️ <b>Trigger Notification:</b> Send automated low-attendance warning to student and academic advisor.</div>', unsafe_allow_html=True)
            if cgpa < 6.0:
                st.markdown('<div class="alert-box" style="border-left-color: #FBBF24;">⚠️ <b>Academic Hold:</b> Schedule mandatory peer-tutoring consultation.</div>', unsafe_allow_html=True)
            if part_time_val == 1 and prob > 0.5:
                st.markdown('<div class="alert-box">⚠️ <b>Schedule Review:</b> Recommend credit-load reduction due to external job commitments.</div>', unsafe_allow_html=True)
            if counsellor == 0:
                st.markdown('<div class="alert-box">⚠️ <b>Outreach:</b> Add to Student Services outreach queue (0 counselor interactions logged).</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    else:
        # Placeholder before submission
        st.markdown("""
        <div style="height: 100%; display: flex; flex-direction: column; justify-content: center; align-items: center; border: 1px dashed #334155; border-radius: 16px; padding: 50px; text-align: center;">
            <div style="font-size: 40px; margin-bottom: 15px;">🔍</div>
            <h3 style="color: #94A3B8; font-weight: 500;">Awaiting Input Data</h3>
            <p style="color: #475569; font-size: 14px;">Fill out the student metrics form and click 'Generate Risk Profile' to view the AI analysis.</p>
        </div>
        """, unsafe_allow_html=True)