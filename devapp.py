import streamlit as st
from tab_home import render_home
from tab_designing import render_designing
from tab_software import render_software
from tab_about import render_about

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="DNA Data Storage Tool",
    page_icon="DDSS_logo.png",
    layout="wide"
)

# --- CUSTOM STYLING (English) ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab"] p { font-size: 24px; font-weight: bold; }
    .stTabs [data-baseweb="tab"] { height: 60px; padding-left: 30px; padding-right: 30px; }
    .stTabs [aria-selected="true"] p { color: #007bff; }
    </style>
    """, unsafe_allow_html=True)

# --- PROFESSIONAL SIDEBAR ---
with st.sidebar:
    st.image("DDSS_logo.png", width=280)
    st.title("DNA Storage Lab")
    st.info("System Version: 1.0.0-Stable")
    st.divider()

    # Infrastructure & Status
    st.subheader("🖥️ System Status")
    st.success("Server Status: Online")
    st.write("**Core Engine:** Connected")
    st.write("**Worker Node:** Active (GPU)")
    
    st.divider()

    # Support & Documentation
    st.subheader("📚 Resources")
    st.button("📖 User Manual", use_container_width=True, help="Detailed guide on encoding/decoding")
    st.button("❓ Help Center", use_container_width=True, help="Frequently asked questions")
    st.button("🛠️ API Documentation", use_container_width=True, help="Integration guides for developers")

    st.divider()

    # Additional Professional Sections
    st.subheader("⚙️ Control Panel")
    st.selectbox("Computing Tier", ["Standard Local", "High-Performance Cloud", "Hybrid Engine"], index=0)
    st.checkbox("Enable Detailed Logs", value=True)
    st.checkbox("Auto-Optimization", value=False)

    st.divider()

    # Professional Copyright Section
    st.markdown("""
        <div style="text-align: center; color: #888; font-size: 15px;">
            <strong>© 2025 DNA Data Storage Lab.</strong><br>
            Sungkyunkwan University.<br>
            All Rights Reserved.
        </div>
    """, unsafe_allow_html=True)

# --- MAIN TAB INTERFACE ---
tab_home, tab_design, tab_soft, tab_about = st.tabs([
    "🏠 Home", 
    "🧬 Designing", 
    "💻 Software", 
    "ℹ️ About Us"
])

with tab_home:
    render_home()

with tab_design:
    render_designing()

with tab_soft:
    render_software()

with tab_about:
    render_about()