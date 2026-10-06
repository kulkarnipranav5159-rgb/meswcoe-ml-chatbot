import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

st.set_page_config(page_title="MESCOE College Portal & AI Chatbot", page_icon="🎓", layout="wide")

# Custom CSS for Floating Chatbot Button in BottomThis error occurs because you have an incomplete assignment statement on line 138 of `app.py`. In Python, an assignment operator (`=`) must be followed by a valid value or expression on the same line or properly formatted across multiple lines.

Here is how to fix it depending on what you were trying to assign to `st.session_state.chat_messages`:

### 1. Assigning an Empty List (Common for Initializing Chat State)
If you intended to initialize the chat history, assign an empty list `[]`:

```python
st.session_state.chat_messages = []
