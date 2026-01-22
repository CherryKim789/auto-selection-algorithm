import streamlit as st

from tab_home import render_home
from tab_designing import render_designing
from tab_software import render_software
from tab_about import render_about

st.set_page_config(
    page_title="DNA Data Storage Tool",
    page_icon="DDSS_logo.png",
    layout="wide"
)

st.markdown("""
    <style>
    div[role="radiogroup"] > label {
        padding: 10px 18px;
        margin-right: 10px;
        border-radius: 10px;
        border: 1px solid rgba(49, 51, 63, 0.2);
        font-size: 18px;
        font-weight: 700;
    }
    div[role="radiogroup"] > label:has(input:checked) {
        border: 2px solid #007bff;
        background: rgba(0, 123, 255, 0.08);
    }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    try:
        st.image("DDSS_logo.png", width=280)
    except Exception:
        pass

    st.title("DNA Storage Lab")
    st.info("System Version: 1.0.0-Stable")
    st.divider()

    st.subheader("🖥️ System Status")
    st.success("Server Status: Online")
    st.write("**Core Engine:** Connected")
    st.write("**Worker Node:** Active")
    st.divider()

    st.subheader("📚 Resources")
    st.button("📖 User Manual", use_container_width=True)
    st.button("❓ Help Center", use_container_width=True)
    st.button("🛠️ API Documentation", use_container_width=True)
    st.divider()

    st.subheader("⚙️ Control Panel")
    st.selectbox("Computing Tier", ["Standard Local", "High-Performance Cloud", "Hybrid Engine"], index=0)
    st.checkbox("Enable Detailed Logs", value=True)
    st.checkbox("Auto-Optimization", value=False)
    st.divider()

    st.markdown("""
        <div style="text-align: center; color: #888; font-size: 13px;">
            <strong>© 2025 DNA Data Storage Lab.</strong><br>
            Sungkyunkwan University.<br>
            All Rights Reserved.
        </div>
    """, unsafe_allow_html=True)

PAGES = {
    "🏠 Home": render_home,
    "🧬 Designing": render_designing,
    "💻 Software": render_software,
    "ℹ️ About Us": render_about,
}

if "main_page" not in st.session_state:
    st.session_state["main_page"] = "🧬 Designing"

page = st.radio(
    "Navigation",
    options=list(PAGES.keys()),
    key="main_page",
    horizontal=True,
    label_visibility="collapsed",
)

PAGES[page]()
