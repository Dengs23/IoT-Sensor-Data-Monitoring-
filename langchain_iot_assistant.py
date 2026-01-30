import os
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timedelta
import numpy as np

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="🤖 LangChain IoT Assistant",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 IoT AI Assistant with LangChain")
st.markdown("Powered by AI for intelligent data analysis")

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.sidebar.warning("🔑 OpenAI API Key Required")
    api_key = st.sidebar.text_input("Enter OpenAI API Key:", type="password")
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.sidebar.success("✅ API Key saved for this session")
        st.rerun()
    else:
        st.info("""
        ## To use this assistant:
        
        1. Get an OpenAI API key from: https://platform.openai.com/api-keys
        2. Enter it in the sidebar
        3. Or create a `.env` file with: `OPENAI_API_KEY=your-key`
        
        **Free alternative:** Use the non-AI version below ⬇️
        """)
        
        if st.button("🚀 Use Non-AI Version Instead"):
            st.switch_page("simple_iot_assistant.py")
        
        st.stop()

# Now import LangChain (after API key check)
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
    
    # Initialize the LLM
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.3,
        api_key=api_key
    )
    
    st.sidebar.success("✅ LangChain loaded successfully!")
    
except ImportError as e:
    st.error(f"❌ LangChain import error: {e}")
    st.info("""
    **To fix this, run in terminal:**
    ```bash
    pip install langchain langchain-openai langchain-core
    ```
    """)
    st.stop()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# System prompt for IoT expertise
system_prompt = SystemMessage(content="""You are an IoT Data and Building Valuation Expert. You specialize in:

1. **IoT Sensor Data Analysis**: Temperature, humidity, battery readings
2. **ROI Calculations**: Return on investment for IoT implementations
3. **Building Valuation**: Property value based on sensor data and risk assessment
4. **Risk Analysis**: Identifying risks from temperature anomalies

You provide:
- Clear, concise answers
- Numerical calculations when possible
- Practical recommendations
- Step-by-step explanations when helpful

Format your responses with clear sections and use bullet points for lists.""")

# Sidebar with example questions
with st.sidebar:
    st.header("💡 Example Questions")
    
    examples = [
        "Analyze the ROI for IoT sensors in a 5000 m² building",
        "How does temperature stability affect building value?",
        "Calculate energy savings from 30% efficiency improvement",
        "What risks do temperature anomalies indicate?",
        "Compare IoT solutions for industrial vs commercial buildings"
    ]
    
    for example in examples:
        if st.button(example, key=f"ex_{hash(example)}"):
            st.session_state.chat_history.append({"role": "user", "content": example})
            st.rerun()
    
    st.header("📊 Quick Tools")
    
    # Quick ROI calculator
    with st.expander("💰 Quick ROI Estimate"):
        size = st.number_input("Building Size (m²)", 1000, 20000, 5000)
        cost = st.number_input("System Cost ($)", 5000, 200000, 75000)
        
        if st.button("Estimate"):
            annual_savings = size * 3
            payback = cost / annual_savings
            st.metric("Annual Savings", f"${annual_savings:,.0f}")
            st.metric("Payback", f"{payback:.1f} years")
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    st.header("💬 AI Conversation")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about IoT data, ROI, or building valuation..."):
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("🤖 AI is thinking..."):
                try:
                    # Create messages for the LLM
                    messages = [
                        system_prompt,
                        HumanMessage(content=prompt)
                    ]
                    
                    # Get response from LLM
                    response = llm.invoke(messages)
                    
                    # Display response
                    st.write(response.content)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": response.content
                    })
                    
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    
                    # Fallback response
                    fallback = f"""I encountered an error but here's what I can tell you about '{prompt}':

For IoT implementations in buildings:
- **Typical ROI**: 150-250% over 10 years
- **Payback Period**: 3-5 years
- **Risk Reduction**: 25-40% with monitoring
- **Value Increase**: 5-15% of property value

Would you like more specific calculations?"""
                    
                    st.write(fallback)
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": fallback
                    })

with col2:
    st.header("📈 Live Data")
    
    # Simulated sensor data
    current_temp = 22.5 + np.random.randn() * 2
    current_humidity = 45 + np.random.randn() * 10
    
    st.metric("🌡️ Temperature", f"{current_temp:.1f}°C")
    st.metric("💧 Humidity", f"{current_humidity:.1f}%")
    st.metric("🔋 Battery", "92%")
    st.metric("📍 Location", "Building A")
    
    # Quick stats
    st.header("📊 Statistics")
    st.metric("Chat Messages", len(st.session_state.chat_history))
    st.metric("AI Model", "GPT-3.5 Turbo")
    st.metric("Response Time", "< 5s")
    
    # Actions
    st.header("⚡ Actions")
    if st.button("🔄 Refresh Data"):
        st.rerun()
    
    if st.button("📥 Export Chat"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iot_chat_export_{timestamp}.txt"
        
        chat_text = "IoT Assistant Chat Export\n"
        chat_text += "=" * 40 + "\n\n"
        
        for msg in st.session_state.chat_history:
            role = "User" if msg["role"] == "user" else "Assistant"
            chat_text += f"{role}: {msg['content']}\n{'─' * 40}\n"
        
        st.download_button(
            label="Download Chat",
            data=chat_text,
            file_name=filename,
            mime="text/plain"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**🤖 LangChain IoT Assistant**")
st.sidebar.markdown("v1.0 | AI-powered analysis")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Auto-suggestions based on chat
if st.session_state.chat_history:
    st.sidebar.markdown("### 🔍 Suggested Next Questions")
    
    suggestions = [
        "What's the maintenance schedule for IoT sensors?",
        "How to integrate with existing building systems?",
        "What data security measures are needed?",
        "Compare wired vs wireless sensor networks"
    ]
    
    for suggestion in suggestions:
        if st.sidebar.button(suggestion, key=f"sugg_{hash(suggestion)}"):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            st.rerun()
