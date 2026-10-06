import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

st.set_page_config(page_title="MESCOE College Portal & AI Chatbot", page_icon="🎓", layout="wide")

# Custom CSS for Floating Chatbot Button in Bottom-Right Corner
st.markdown("""
    <style>
        /* Floating Chatbot Container */
        div[data-testid="stPopover"] {
            position: fixed !important;
            bottom: 30px !important;
            right: 30px !important;
            z-index: 999999 !important;
        }
        /* Style the Floating Trigger Button */
        div[data-testid="stPopover"] > button {
            background-color: #1e3a8a !important;
            color: white !important;
            border-radius: 50px !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border: 2px solid white !important;
            box-shadow: 0px 4px 15px rgba(0,0,0,0.3) !important;
        }
        div[data-testid="stPopover"] > button:hover {
            background-color: #2563eb !important;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Load model artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('mescoe_chatbot_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    
    try:
        df = pd.read_csv('mescoe_dataset_expanded.csv')
    except Exception:
        df = pd.read_csv('mescoe_dataset.csv')
        
    metrics_df = pd.read_csv('model_metrics.csv')
    return model, vectorizer, df, metrics_df

try:
    model, vectorizer, dataset, metrics_df = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}")
    st.stop()

# Clean dictionary mapping intent string to its response
intent_response_map = dict(zip(dataset['intent'], dataset['response']))

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🏛️ MESCOE Website & AI Chatbot", "📊 Model Metrics", "📁 Dataset Preview"])

# TAB 1: DUMMY COLLEGE WEBSITE & FLOATING ML CHATBOT
with tab1:
    # Render Full College Website UI
    website_html = r'''
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: #f4f6f9; }
        header { background: #1e3a8a; color: white; padding: 18px 40px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { margin: 0; font-size: 22px; }
        nav a { color: white; margin-left: 20px; text-decoration: none; font-weight: 500; font-size: 15px; }
        .hero { background: linear-gradient(rgba(30,58,138,0.85), rgba(30,58,138,0.85)), url('https://mescoe.mespune.org/wp-content/uploads/2021/09/slider1.jpg'); 
                background-size: cover; color: white; padding: 60px 20px; text-align: center; }
        .hero h2 { font-size: 32px; margin-bottom: 10px; }
        .hero p { font-size: 18px; opacity: 0.9; }
        .cards { display: flex; gap: 20px; padding: 40px 20px; justify-content: center; flex-wrap: wrap; }
        .card { background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 200px; text-align: center; border-top: 4px solid #1e3a8a; }
        .card h3 { color: #1e3a8a; margin-top: 0; font-size: 18px; }
        .card p { font-size: 14px; color: #475569; margin-top: 10px; line-height: 1.5; }
        footer { background: #0f172a; color: #94a3b8; text-align: center; padding: 20px; font-size: 14px; margin-top: 20px; }
    </style>
    </head>
    <body>

    <header>
        <h1>Modern Education Society's Wadia College of Engineering</h1>
        <nav>
            <a href="#">Home</a>
            <a href="#">Admissions</a>
            <a href="#">Departments</a>
            <a href="#">Placements</a>
        </nav>
    </header>

    <div class="hero">
        <h2>Welcome to MESCOE Pune</h2>
        <p>NAAC 'A++' Grade Accredited Institute | SPPU Affiliated</p>
    </div>

    <div class="cards">
        <div class="card">
            <h3>Computer Engg</h3>
            <p>UG Intake: 300<br>PG Intake: 12</p>
        </div>
        <div class="card">
            <h3>E & TC Engg</h3>
            <p>UG Intake: 120<br>PG Intake: 12</p>
        </div>
        <div class="card">
            <h3>Mechanical Engg</h3>
            <p>UG Intake: 60<br>PG Intake: 12</p>
        </div>
        <div class="card">
            <h3>Automation & Robotics</h3>
            <p>UG Intake: 60</p>
        </div>
    </div>

    <footer>
        <p>© Modern Education Society's Wadia College of Engineering, Pune. All Rights Reserved.</p>
    </footer>

    </body>
    </html>
    '''
    components.html(website_html, height=520, scrolling=True)

    # FLOATING AI CHATBOT WIDGET
    with st.popover("💬 Chat with AI Assistant"):
        st.subheader("🤖 MESCOE AI Chatbot")
        st.caption("Ask questions about courses, admissions, fees, hostel, or placements.")

        if "chat_messages" not in st.session_state:
            st.session_state.chat_messages =
