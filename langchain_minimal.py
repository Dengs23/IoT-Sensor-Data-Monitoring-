import os
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.title("🤖 LangChain IoT Assistant")

# Initialize the LLM
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Chat interface
prompt = st.text_input("Ask about IoT or ROI:")

if prompt:
    with st.spinner("Thinking..."):
        response = llm.invoke([
            HumanMessage(content=f"""You are an IoT data expert. Help with: 
            1. Sensor data analysis
            2. ROI calculations for IoT systems
            3. Building valuation based on temperature data
            4. Risk assessment
            
            User question: {prompt}
            
            Provide a helpful, concise answer.""")
        ])
        
    st.write("🤖 Assistant:", response.content)
