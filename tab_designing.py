import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

def render_designing():
    st.header("🧬 DNA Data Design & Pipeline Management")
    st.markdown("Manage the complete lifecycle of DNA data storage, from digital encoding to retrieval.")

    # Create 4 Sub-tabs inside the Designing Tab
    sub_tab_encoding, sub_tab_wetlab, sub_tab_decoding, sub_tab_analysis = st.tabs([
        "📤 1. Encoding & Design", 
        "🧪 2. Wet-lab Simulation", 
        "📥 3. Decoding & Retrieval", 
        "🔬 4. Comparative Analysis"
    ])

    # ---------------------------------------------------------
    # SUB-TAB 1: ENCODING & DESIGN
    # ---------------------------------------------------------
    with sub_tab_encoding:
        st.subheader("DNA Encoding Pipeline")
        st.markdown("Transform digital files into synthesis-ready DNA sequences.")

        st.subheader("1. Data Binarization")
        col1, col2 = st.columns([2, 1])
        with col1:
            uploaded_file = st.file_uploader("Select Input File (Image, Video, Document)", 
                                             type=['png', 'jpg', 'pdf', 'mp4', 'txt'], 
                                             key="design_upload")
        if uploaded_file:
            with col2:
                st.success(f"File Loaded: {uploaded_file.name}")
                st.info(f"Size: {uploaded_file.size/1024:.2f} KB")
            
            st.divider()
            st.subheader("2. Binary Compression")
            c_comp1, c_comp2 = st.columns(2)
            with c_comp1:
                compression_algo = st.selectbox("Compression Method", ["None", "LZMA", "Gzip", "Zstd"], key="comp_algo")
            with c_comp2:
                st.write("**Target Binary String:**")
                st.code("01011010... (Compressed)", language="text")

            st.divider()
            st.subheader("3. Sequence Design & Mapping")
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                mapping_rule = st.selectbox("Encoding Rule", ["Simple (00=A, 01=C...)", "Rotational (R1)", "Church et al."], key="map_rule")
            with m_col2:
                fec_type = st.selectbox("Error Correction (FEC)", ["Reed-Solomon", "LDPC", "Fountain Code"], key="fec_design")
            with m_col3:
                redundancy = st.slider("Redundancy (%)", 0, 50, 15, key="red_design")

            with st.expander("⚙️ Advanced Biological Constraints"):
                gc_range = st.slider("Target GC Content (%)", 30, 70, (40, 60))
                homo_limit = st.number_input("Max Homopolymer Length", value=3)

            st.divider()
            if st.button("🚀 EXECUTE ENCODING PIPELINE", key="btn_execute_design"):
                st.balloons()
                st.subheader("📊 Encoding Performance")
                met1, met2, met3 = st.columns(3)
                met1.metric("Density", "1.65 bits/nt")
                met2.metric("Efficiency", "98.2%")
                met3.metric("Net Rate", "0.85")

                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    df_base = pd.DataFrame({'Nucleotide': ['A', 'C', 'G', 'T'], 'Ratio (%)': [24.8, 25.2, 25.1, 24.9]})
                    fig_base = px.pie(df_base, values='Ratio (%)', names='Nucleotide', title="Nucleotide Distribution")
                    st.plotly_chart(fig_base, use_container_width=True)
                with chart_col2:
                    gc_data = np.random.normal(50, 2, 100)
                    fig_gc = px.line(y=gc_data, title="GC Content Stability")
                    st.plotly_chart(fig_gc, use_container_width=True)

                st.subheader("🔬 Final DNA Pool")
                st.text_area("FASTA Preview", value=">Segment_01\nATCGGCTAGCT...\n>Segment_02\nGCTAGCTAGCT...", height=100)
                st.download_button("📥 Download FASTA", data=">DNA_DATA", file_name="output.fasta")
        else:
            st.info("Please upload a file to begin.")

    # ---------------------------------------------------------
    # SUB-TAB 2: WET-LAB SIMULATION
    # ---------------------------------------------------------
    with sub_tab_wetlab:
        st.subheader("🧪 Wet-lab & Noise Simulation")
        st.markdown("Simulate the chemical and biological processes that introduce errors.")
        
        col_w1, col_w2 = st.columns(2)
        with col_w1:
            st.write("**Synthesis & PCR Parameters**")
            st.slider("Synthesis Error Rate (per nt)", 0.0, 0.05, 0.01, format="%.3f")
            st.number_input("PCR Cycles", 10, 40, 25)
        with col_w2:
            st.write("**Storage Conditions**")
            st.select_slider("Storage Time Simulation", options=["1 Year", "10 Years", "100 Years", "1000 Years"])
            st.slider("Decay Factor", 0.0, 1.0, 0.05)
        
        if st.button("🧪 Run Noise Simulation", key="btn_sim"):
            st.warning("Simulating data degradation...")
            st.info("Resulting BER (Bit Error Rate): 0.045% after simulation.")

    # ---------------------------------------------------------
    # SUB-TAB 3: DECODING
    # ---------------------------------------------------------
    with sub_tab_decoding:
        st.subheader("📥 Decoding & Data Retrieval")
        st.markdown("Reconstruct the original digital file from sequenced DNA reads.")
        
        uploaded_fastq = st.file_uploader("Upload Sequenced Data (FASTQ/FASTA)", type=['fastq', 'fasta', 'txt'], key="decode_upload")
        
        if uploaded_fastq:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.write("**Decoding Configuration**")
                st.selectbox("Inference Algorithm", ["Viterbi", "BWA-MEM Alignment", "Deep Learning-based"], key="algo_dec")
                st.checkbox("Apply FEC Recovery", value=True)
            
            with col_d2:
                st.write("**Quality Control (QC)**")
                st.metric("Mean Phred Score", "34.2")
                st.progress(95, text="Alignment Progress")

            if st.button("🔓 Start Decoding", key="btn_decode"):
                st.spinner("Reconstructing file...")
                st.success("File Successfully Restored!")
                st.download_button("📥 Download Restored File", data="dummy_data", file_name="restored_file.png")
        else:
            st.info("Please upload sequenced DNA data to start decoding.")

    # ---------------------------------------------------------
    # SUB-TAB 4: ANALYSIS
    # ---------------------------------------------------------
    with sub_tab_analysis:
        st.subheader("🔬 Comparative Analysis")
        st.write("Evaluate the performance of different encoding/decoding strategies.")
        
        # Mock Data for Analysis
        df_analysis = pd.DataFrame({
            'Method': ['Simple', 'Rotational', 'Church', 'Our Method'],
            'PSNR (dB)': [32.1, 35.4, 34.0, 38.5],
            'Bit Error Rate': [0.02, 0.012, 0.015, 0.004]
        })
        
        col_an1, col_an2 = st.columns(2)
        with col_an1:
            fig_psnr = px.bar(df_analysis, x='Method', y='PSNR (dB)', title="Visual Quality (PSNR)")
            st.plotly_chart(fig_psnr, use_container_width=True)
        with col_an2:
            fig_ber = px.line(df_analysis, x='Method', y='Bit Error Rate', title="Reliability (BER)")
            st.plotly_chart(fig_ber, use_container_width=True)