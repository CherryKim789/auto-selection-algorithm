import streamlit as st

def render_about():
    st.header("DNA Data Storage System")
    st.markdown("""
    - **Objective:** Develop end-to-end system with a high-density, long-term DNA data storage method with robust encoding, error correction, and reliable data retrieval.
    - Biological information in DNA is stored as a code recorded in the sequential ordering of the four chemical bases. The order, or sequence, of these bases determines the information available for building and maintaining an organism, similar to the way in which letters of the alphabet appear in a certain order to form words and sentences. Consequently, DNA is treated as a promising data storage medium due to its stability and information storage capacity. All available data can be stored in DNA sequences by mapping the digital information into DNA base sequences. We develop computer codes with error correction scheme that encode digital information into DNA base sequences and decodes the information back from DNA base sequences to a digital file.
    """)
    st.header("Members and Contact")
    st.write("**If you have any questions or feedback, please feel free to contact us:**")
    st.markdown("""
    <div class="contact-card">
        <strong>Prof. Sung Ha Park</strong><br>
        BIG BOSS<br>
        Email: sunghapark@skku.edu<br>
        Phone: 010-xxxx-xxxx<br>
    </div>          
    <div class="contact-card">
        <strong>Dr. Dinosaur N.K.U</strong><br>
        App Developer<br>
        Email: kimuyendlu@gmail.com<br>
        Phone: 010-xxxx-xxxx<br>
    </div>
    """, unsafe_allow_html=True)

    st.header("Location")
    st.markdown("""
    - **Address:** Department of Physics, Sungkyunkwan University,
                2066, Seobu-ro, Jangan-gu, Suwon, Gyeonggi-do, 16419,
                Republic of Korea
    """)
    map_html = """
<iframe src="https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3171.303975005891!2d126.97210167637823!3d37.2938883394593!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x357b56b21761867f%3A0xb38ea754e92d9bb0!2sSungkyunkwan%20University%20(Natural%20Sciences%20Campus)!5e0!3m2!1sen!2skr!4v1700000000000!5m2!1sen!2skr" 
    width="100%" 
    height="450" 
    style="border:0; border-radius: 10px;" 
    allowfullscreen="" 
    loading="lazy" 
    referrerpolicy="no-referrer-when-downgrade">
</iframe>
"""
    st.components.v1.html(map_html, height=500)