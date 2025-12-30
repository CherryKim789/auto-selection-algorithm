import streamlit as st

def render_software():
    # --- PHẦN 1: DOWNLOAD SOFTWARE ---
    st.header("💾 Download Software")
    st.write("""
    **Our software is developed in-house and provided for offline use, enabling reliable operation 
    in internet-restricted environments such as cleanrooms, or whenever users require stable, 
    uninterrupted performance independent of network connectivity.**
    """)
    # Tạo 3 cột để hiển thị 3 phiên bản tải về
    col_v1, col_v2, col_v3 = st.columns(3)
    
    with col_v1:
        st.subheader("DDSS ver.01")
        st.caption("Release Date: 2025.11")
        st.button("📥 Download v01", key="dl_v1", help="Stable version for basic encoding")

    with col_v2:
        st.subheader("DDSS ver.02")
        st.caption("Release Date: 2025.12")
        st.button("📥 Download v02", key="dl_v2", help="Added Reed-Solomon support")

    with col_v3:
        st.subheader("DDSS ver.03")
        st.caption("Latest Version")
        st.button("🚀 Download v03", key="dl_v3", help="Optimized for large video files")

    st.write("""
    If you need any assistance or encounter any issues, please feel free to contact us.
    """)
    st.divider()
    # --- PHẦN 2: RELATED TOOLS ---
    st.header("🔗 Related Tools")
    st.write("Below are the essential supporting tools and platforms for the research and development workflow:")
    # Sử dụng cột để trình bày các link liên kết cho gọn gàng
    link_col1, link_col2, link_col3 = st.columns(3)

    with link_col1:
        st.markdown("""
        **Programming & Environment**
        - [Python Official](https://www.python.org/)
        - [Anaconda Distribution](https://www.anaconda.com/)
        - [Visual Studio Code](https://code.visualstudio.com/)
        """)

    with link_col2:
        st.markdown("""
        **AI & Support**
        - [ChatGPT (OpenAI)](https://chatgpt.com/)
        - [Claude AI](https://claude.ai/)
        - [GitHub Copilot](https://github.com/features/copilot)
        """)

    with link_col3:
        st.markdown("""
        **Bioinformatics Resources**
        - [Biopython Project](https://biopython.org/)
        - [NCBI Database](https://www.ncbi.nlm.nih.gov/)
        - [Hugging Face](https://huggingface.co/)
        """)



    st.button("Check for Updates")
