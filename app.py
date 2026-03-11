import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt

# 1. Page Identity
st.set_page_config(page_title="Talking Rabbitt MVP", page_icon="🐰")
st.title("🐰 Talking Rabbitt")
st.subheader("Conversational Intelligence Layer")

# 2. Sidebar - Configuration
with st.sidebar:
    st.write("### ⚙️ Configuration")
    api_key = st.text_input("OpenAI API Key", type="password")
    uploaded_file = st.file_uploader("Upload Sales CSV", type=['csv'])

# 3. Execution Logic
if uploaded_file and api_key:
    # Load data
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Data Preview (Top 5 Rows)")
    st.dataframe(df.head())

    # Initialize Talking Rabbitt Engine
    llm = OpenAI(api_token=api_key)
    # SmartDataframe makes the pandas df 'conversational'
    agent = SmartDataframe(df, config={"llm": llm})

    # 4. The "Magic Moment" - Query Input
    query = st.text_input("Ask a question about your business:")

    if query:
        with st.spinner("Rabbitt is thinking..."):
            response = agent.chat(query)
            
            # Display results
            st.write("### 💡 Insight")
            st.success(response)
            
            # Automated Visualization Check
            # PandasAI saves the last generated plot to 'exports/charts'
            # Or you can force it to display here:
            fig = plt.gcf() 
            if fig.get_axes(): # Check if a plot was actually created
                st.pyplot(fig)
else:
    st.info("Please enter your API Key and upload a CSV to begin.")