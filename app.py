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
except Exception as e:
    st.error(f"Error loading model assets: {e}")
    st.stop()

intent_response_map = dict(zip(dataset['intent'], dataset['response']))

# Create tabs (Hidden sidebar project details removed)
tab1, tab2, tab3 = st.tabs(["🏛️ MESCOE Website & AI Chatbot", "📊 Model Metrics", "📁 Dataset Preview"])

# TAB 1: DUMMY COLLEGE WEBSITE & ML CHATBOT
with tab1:
    # Render College Website Banner & Header HTML
    website_html = """
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; background: #f4f6f9; }
        header { background: #1e3a8a; color: white; padding: 15px 30px; display: flex; justify-content: space-between; align-items: center; }
        header h1 { margin: 0; font-size: 20px; }
        nav a { color: white; margin-left: 15px; text-decoration: none; font-weight: 500; font-size: 14px; }
        .hero { background: linear-gradient(rgba(30,58,138,0.85), rgba(30,58,138,0.85)), url('https://mescoe.mespune.org/wp-content/uploads/2021/09/slider1.jpg'); 
                background-size: cover; color: white; padding: 40px 20px; text-align: center; }
        .cards { display: flex; gap: 15px; padding: 20px; justify-content: center; flex-wrap: wrap; }
        .card { background: white; padding: 15px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); width: 180px; text-align: center; }
        .card h3 { color: #1e3a8a; margin-top: 0; font-size: 16px; }
        .card p { font-size: 13px; color: #333; margin: 5px 0 0 0; }
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

    </body>
    </html>
    """
    components.html(website_html, height=280, scrolling=False)

    st.markdown("---")
    st.subheader("💬 MESCOE AI Assistant")
    st.caption("Ask questions about courses, admissions, fees, hostel, exams, or placements.")

    # Initialize chat memory
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Hello! Welcome to MES Wadia College of Engineering. How can I help you today?"}
        ]

    # Display chat log
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Interactive Chat Input & ML Intent Classification
    if user_input := st.chat_input("Type your question here (e.g., What courses are available?)..."):
        st.session_state.chat_messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Scikit-learn TF-IDF Vectorization & Prediction
        input_vector = vectorizer.transform([user_input])
        predicted_intent = model.predict(input_vector)[0]
        bot_response = intent_response_map.get(
            predicted_intent, 
            "I apologize, I didn't quite understand that. Please contact info@mescoepune.org for details."
        )

        with st.chat_message("assistant"):
            st.write(bot_response)
            st.caption(f"*ML Predicted Intent:* `{predicted_intent}`")

        st.session_state.chat_messages.append({"role": "assistant", "content": bot_response})

# TAB 2: MODEL EVALUATION
with tab2:
    st.header("📊 Model Metrics")
    st.dataframe(metrics_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.barplot(x='Model', y='Accuracy', data=metrics_df, palette='Blues_d', ax=ax)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy")
    st.pyplot(fig)

# TAB 3: DATASET PREVIEW
with tab3:
    st.header("📁 Dataset Preview")
    st.dataframe(dataset, use_container_width=True)
