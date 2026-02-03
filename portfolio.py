import streamlit as st
from pathlib import Path

st.set_page_config(page_title="IoT Portfolio", layout="wide")
st.title("🏢 IoT Dashboard Portfolio")

# Count apps
prod_files = list(Path("apps/production").glob("*.py"))
dev_files = list(Path("apps/development").glob("*.py"))
arch_files = list(Path("apps/archived").glob("*.py"))

st.write(f"**📊 Production Apps:** {len(prod_files)}")
st.write(f"**🔧 Development Apps:** {len(dev_files)}")
st.write(f"**📦 Archived Apps:** {len(arch_files)}")
st.success(f"**🎯 Total:** {len(prod_files) + len(dev_files) + len(arch_files)} Python applications")

# Show main apps
st.write("### 🚀 Main Dashboards:")
for app in ["iot_dash_advanced.py", "iot_dash_complete.py", "complex_iot_assistant_app.py"]:
    st.write(f"- `{app}`")

st.write("### 💻 How to Run:")
st.code("""
# Run this portfolio:
streamlit run portfolio.py

# Run main dashboard:
python apps/production/iot_dash_advanced.py

# Access at:
# Portfolio: http://localhost:8501
# Dashboard: http://localhost:8050
""")
