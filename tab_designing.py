import streamlit as st
import pandas as pd
import binascii
import lzma
import io
import os
import numpy as np
# Import functions from your uploaded codec file
from dna_codec import encode_bits_to_dna, gc_content, homopolymer_stats, clean_dna_text

def get_file_details(uploaded_file):
    """Fetch file metadata for the left sidebar display."""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    details = {
        "Filename": uploaded_file.name,
        "MIME Type": uploaded_file.type,
        "Extension": ext.upper(),
        "Size": f"{uploaded_file.size:,} bytes"
    }
    return details, ext

def process_to_bitstream(uploaded_file, ext):
    """Convert various file types to bitstream, handling raw binary text files."""
    content = uploaded_file.getvalue()
    # Check if the file is already a text bitstream (0s and 1s)
    if ext in ['.txt', '.bin']:
        try:
            decoded_text = content.decode("utf-8").strip()
            if all(c in '01' for c in decoded_text):
                return decoded_text, "Direct Bitstream Read"
        except:
            pass
    
    # Standard conversion for images, docs, etc.
    bitstream = bin(int(binascii.hexlify(content), 16))[2:].zfill(len(content) * 8)
    return bitstream, "Hex-to-Binary Conversion"

def generate_fastq(dna_seq):
    """Simulate a FASTQ format with high-quality Phred scores."""
    header = "@DNA_STORAGE_STREAM_01"
    quality = "I" * len(dna_seq) # 'I' represents a high Phred score of 40
    return f"{header}\n{dna_seq}\n+\n{quality}"

def display_preview(uploaded_file, ext):
    """Visual preview of the input data."""
    if ext in ['.png', '.jpg', '.jpeg']:
        st.image(uploaded_file, use_container_width=True)
    elif ext in ['.txt', '.py', '.dna']:
        st.text_area("File Content Preview", uploaded_file.getvalue().decode("utf-8")[:500], height=200)
    elif "video" in uploaded_file.type:
        st.video(uploaded_file)
    else:
        st.info("No visual preview available for this file format.")

def render_designing():
    st.header("🧬 DNA Data Design & Pipeline Management")
    
    tab_raw, tab_comp = st.tabs(["📄 Case 1: Raw Data Pipeline", "📦 Case 2: Compressed Data Pipeline"])

    mapping_rules = ["R0_B9", "R1_B12", "R2_B15", "Rinf_B16", "Simple Mapping"]

    # --- CASE 1: RAW DATA ---
    with tab_raw:
        uploaded_file = st.file_uploader("Upload Data (Images, Docs, or Bitstream .txt)", key="raw_up")
        if uploaded_file:
            col_l, col_r = st.columns([1, 2])
            details, ext = get_file_details(uploaded_file)
            
            with col_l:
                st.subheader("Data Metadata")
                for k, v in details.items(): st.write(f"**{k}:** {v}")
                rule = st.selectbox("Select Mapping Rule", mapping_rules, key="r1")
            
            with col_r:
                st.subheader("Visual Preview")
                display_preview(uploaded_file, ext)

            if st.button("🚀 Execute Raw Pipeline", key="exec_raw"):
                bits, method = process_to_bitstream(uploaded_file, ext)
                dna_seq, _ = encode_bits_to_dna(bits, scheme_name=rule if rule != "Simple Mapping" else "R1_B12")
                
                # --- RESULTS ---
                st.divider()
                st.success(f"Encoding Complete using {method}")
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.write("**Bitstream Preview (Top 128 bits):**")
                    st.code(bits[:128] + "...")
                    st.download_button("📥 Download Bitstream (.bin)", data=bits, file_name="raw_bitstream.bin")
                
                with res_col2:
                    st.write(f"**DNA Sequence ({rule}):**")
                    st.code(dna_seq[:100] + "...")
                    st.download_button("📥 Download DNA Text (.dna)", data=dna_seq, file_name="output.dna")

                st.subheader("Analysis & Export")
                m1, m2, m3 = st.columns(3)
                m1.metric("GC Content", f"{gc_content(dna_seq):.2%}")
                stats = homopolymer_stats(dna_seq)
                m2.metric("Max Homopolymer", stats['longest'])
                m3.metric("Density", f"{len(bits)/len(dna_seq):.2f} bits/nt")

                dl1, dl2 = st.columns(2)
                dl1.download_button("📥 Download FASTA", data=f">RAW_DATA_{rule}\n{dna_seq}", file_name="data.fasta")
                dl2.download_button("📥 Download FASTQ", data=generate_fastq(dna_seq), file_name="data.fastq")

    # --- CASE 2: COMPRESSED DATA ---
    with tab_comp:
        uploaded_file_c = st.file_uploader("Upload Data for Compression", key="comp_up")
        if uploaded_file_c:
            col_l, col_r = st.columns([1, 2])
            details, ext = get_file_details(uploaded_file_c)
            
            with col_l:
                st.subheader("Data Metadata")
                for k, v in details.items(): st.write(f"**{k}:** {v}")
                rule_c = st.selectbox("Select Mapping Rule", mapping_rules, key="r2")
            
            with col_r:
                st.subheader("Visual Preview")
                display_preview(uploaded_file_c, ext)

            if st.button("🚀 Execute Compressed Pipeline", key="exec_comp"):
                # Compression Step
                original_bytes = uploaded_file_c.getvalue()
                compressed_bytes = lzma.compress(original_bytes)
                ratio = (1 - (len(compressed_bytes)/len(original_bytes))) * 100
                
                st.warning(f"Note: Data compressed using LZMA algorithm. Compression Ratio: {ratio:.2f}%")
                
                # Convert compressed bytes to bitstream
                c_bits = bin(int(binascii.hexlify(compressed_bytes), 16))[2:].zfill(len(compressed_bytes) * 8)
                dna_seq, _ = encode_bits_to_dna(c_bits, scheme_name=rule_c if rule_c != "Simple Mapping" else "R1_B12")
                
                st.divider()
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.write("**Compressed Bitstream:**")
                    st.code(c_bits[:128] + "...")
                    st.download_button("📥 Download Comp. Bitstream", data=c_bits, file_name="comp_bitstream.bin")
                
                with res_col2:
                    st.write(f"**DNA Sequence ({rule_c}):**")
                    st.code(dna_seq[:100] + "...")
                    st.download_button("📥 Download DNA Sequence", data=dna_seq, file_name="comp_output.txt")

                st.subheader("Analysis & Scientific Metrics")
                stats = homopolymer_stats(dna_seq)
                st.json({
                    "Compression_Ratio": f"{ratio:.2f}%",
                    "GC_Content": f"{gc_content(dna_seq):.2%}",
                    "Homopolymer_Stats": stats,
                    "Total_Nucleotides": len(dna_seq)
                })

                dl_a, dl_b = st.columns(2)
                dl_a.download_button("📥 Download FASTA (Comp)", data=f">COMP_DATA_{rule_c}\n{dna_seq}", file_name="comp.fasta")
                dl_b.download_button("📥 Download FASTQ (Comp)", data=generate_fastq(dna_seq), file_name="comp.fastq")