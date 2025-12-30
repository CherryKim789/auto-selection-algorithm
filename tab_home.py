import streamlit as st

def render_home():
    st.title("Welcome to DNA Data Storage Project")
    
    # Overview Section
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Exploring the Future of Data Storage
        This system allows you to transform multimedia files into DNA sequences 
        optimized for chemical synthesis in a wet-lab environment.
        
        **Key Process:**
        1. **Encoding:** Convert Binary data into DNA Mapping Rules.
        2. **Optimization:** Verify biological stability (GC content, Homopolymers).
        3. **Decoding:** Restore original data from sequencing results.
        """)
    with col2:
        # Displaying the DNA Structure as a visual aid
        st.image("endtoend.jpg", 
                 caption="DNA Data Storage: End-to-end System Overview")

    st.divider()

    # Workflow Section - Adding the workflow.jpg here
    st.header("🧬 Indicated Workflow")
    st.markdown("The following diagram illustrates the end-to-end pipeline from digital data to DNA synthesis and back.")
    
    # Căn giữa hình ảnh workflow
    col_flow, _ = st.columns([10, 1])
    with col_flow:
        st.image("workflow.jpg", 
                 caption="End-to-End DNA Data Storage Workflow: Encoding, Wet-lab, and Decoding", 
                 use_container_width=True)
    
    st.info("**Note:** This workflow ensures robust data retrieval through advanced error correction (FEC) and biological constraint mapping.")