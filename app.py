import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

st.set_page_config(page_title="MESCOE College Portal & AI Chatbot", page_icon="🎓", layout="wide")

# Load model artifacts
@st.cache_resource
def load_assets():
    model = joblib.load('mescoe_chatbot_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    df = pd.read_csv('mescoe_dataset.csv')
    metrics_df = pd.read_csv('model_metrics.csv')
    return model, vectorizer, df, metrics_df

try:
    model, vectorizer, dataset, metrics_df = load_assets()
except Exception:
    st.error("Please run 'python train_model.py' first!")
    st.stop()

intent_response_map = dict(zip(dataset['intent'], dataset['response']))

# Sidebar Configuration
st.sidebar.title("🎓 Project Details")
st.sidebar.info("""
**TE Machine Learning Mini Project**
* **Project:** College Dummy Portal + ML Chatbot
* **College:** MES Wadia College of Engineering (MESCOE)
""")
st.sidebar.markdown("---")
st.sidebar.subheader("👥 Team Members")
st.sidebar.write("1. Student Name 1")
st.sidebar.write("2. Student Name 2")
st.sidebar.write("3. Student Name 3")

# Main Navigation Tabs
tab1, tab2, tab3 = st.tabs(["🏛️ MESCOE Website & AI Chatbot", "📊 ML Model Metrics", "📁 Training Dataset"])

# TAB 1: DUMMY COLLEGE WEBSITE WITH FLOATING CHATBOT WIDGET
with tab1:
    st.subheader("MES Wadia College of Engineering - Virtual Web Portal")
    
    # Process user query from embedded UI widget if posted
    query_params = st.query_params
    user_query = query_params.get("chat_query", "")
    bot_reply = ""
    predicted_intent = ""

    if user_query:
        input_vec = vectorizer.transform([user_query])
        predicted_intent = model.predict(input_vec)[0]
        bot_reply = intent_response_map.get(predicted_intent, "For details, please email info@mescoepune.org.")

    # Embedded HTML/CSS UI for College Replica & Widget
    website_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: #f4f6f9; }}
        header {{ background: #1e3a8a; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }}
        header h1 {{ margin: 0; font-size: 22px; }}
        nav a {{ color: white; margin-left: 20px; text-decoration: none; font-weight: 500; }}
        .hero {{ background: linear-gradient(rgba(30,58,138,0.8), rgba(30,58,138,0.8)), url('https://mescoe.mespune.org/wp-content/uploads/2021/09/slider1.jpg'); 
                background-size: cover; color: white; padding: 60px 30px; text-align: center; }}
        .cards {{ display: flex; gap: 20px; padding: 30px; justify-content: center; flex-wrap: wrap; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 220px; text-align: center; }}
        .card h3 {{ color: #1e3a8a; margin-top: 0; }}
        
        /* Floating Chat Widget */
        .chat-btn {{ position: fixed; bottom: 20px; right: 20px; background: #1e3a8a; color: white; border: none; 
                    padding: 15px 22px; border-radius: 30px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 10px rgba(0,0,0,0.3); }}
        .chat-box {{ position: fixed; bottom: 80px; right: 20px; width: 340px; height: 420px; background: white; 
                    border-radius: 12px; box-shadow: 0 5px 20px rgba(0,0,0,0.2); display: flex; flex-direction: column; overflow: hidden; }}
        .chat-header {{ background: #1e3a8a; color: white; padding: 12px; font-weight: bold; text-align: center; }}
        .chat-body {{ flex: 1; padding: 15px; overflow-y: auto; font-size: 14px; display: flex; flex-direction: column; gap: 10px; }}
        .msg {{ padding: 8px 12px; border-radius: 8px; max-width: 80%; }}
        .bot {{ background: #e2e8f0; color: #1e293b; align-self: flex-start; }}
        .user {{ background: #1e3a8a; color: white; align-self: flex-end; }}
        .chat-footer {{ display: flex; padding: 10px; border-top: 1px solid #ddd; }}
        .chat-footer input {{ flex: 1; padding: 8px; border: 1px solid #ccc; border-radius: 4px; }}
        .chat-footer button {{ background: #1e3a8a; color: white; border: none; padding: 8px 12px; margin-left: 5px; border-radius: 4px; cursor: pointer; }}
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

    <!-- AI Chatbot Box -->
    <div class="chat-box" id="chatBox">
        <div class="chat-header">🤖 MESCOE AI Assistant</div>
        <div class="chat-body" id="chatBody">
            <div class="msg bot">Hello! Welcome to MESCOE Wadia College. How can I help you today?</div>
            {f'<div class="msg user">{user_query}</div><div class="msg bot">{bot_reply}<br><br><small style="color:#64748b;">(Predicted Intent: {predicted_intent})</small></div>' if user_query else ''}
        </div>
        <form class="chat-footer" action="" method="get" target="_top">
            <input type="text" name="chat_query" placeholder="Ask about fees, courses..." required />
            <button type="submit">Send</button>
        </form>
    </div>

    </body>
    </html>
    """
    
    # Render embedded Portal inside Streamlit
    components.html(website_html, height=620, scrolling=True)

# TAB 2: MODEL EVALUATION & METRICS
with tab2:
    st.header("📈 Machine Learning Performance & Evaluation")
    st.write("Comparison metrics between **Multinomial Naive Bayes** and **Logistic Regression**.")
    
    st.dataframe(metrics_df, use_container_width=True)
    
    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x='Model', y='Accuracy', data=metrics_df, palette='Blues_d', ax=ax)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy Score")
    ax.set_title("Classifier Model Accuracy Comparison")
    st.pyplot(fig)

# TAB 3: DATASET PREVIEW
with tab3:
    st.header("📄 MESCOE Labeled Dataset")
    st.dataframe(dataset, use_container_width=True)