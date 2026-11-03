"""Tab 3: Konsultasi & Tanya Jawab AI (Agri-Chatbot)."""

import streamlit as st
from constants import CLASS_NAMES_ID
from chatbot import get_chat_response


def render_tab_chatbot():
    """Merender Tab 3 untuk konsultasi agronomi interaktif berbasis AI."""
    st.markdown('<div class="section-header">Asisten Pakar Padi & Konsultasi AI</div>', unsafe_allow_html=True)
    st.write(
        "Tanyakan apa saja seputar penyakit daun padi, langkah penanganan hayati/kimiawi, "
        "rekomendasi takaran pupuk berimbang, pemilihan varietas tahan, maupun aspek teknis model AI skripsi."
    )

    # Indikator Konteks Penyakit yang Terdeteksi
    active_disease = st.session_state.get("active_disease", None)
    active_conf = st.session_state.get("active_conf", None)

    if active_disease and active_disease in CLASS_NAMES_ID:
        st.markdown(
            f"""
            <div style="background: #eaf5e9; border: 1px solid #b7deb3; border-radius: 10px; padding: .65rem 1rem; margin-bottom: 1rem; display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <span style="font-size: .8rem; text-transform: uppercase; font-weight: 700; color: #2e7d58;">Konteks Diagnosa Aktif:</span>
                    <strong style="color: #142a22; margin-left: .4rem;">{CLASS_NAMES_ID[active_disease]}</strong>
                    <span style="color: #566961; font-size: .85rem; margin-left: .3rem;">({active_conf:.2%})</span>
                </div>
                <span style="background: #2e7d58; color: white; border-radius: 999px; padding: .2rem .6rem; font-size: .72rem; font-weight: 700;">TERHUBUNG</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("ℹ️ *Belum ada citra yang dideteksi pada Tab 1. Asisten AI akan menjawab dalam mode konsultasi umum.*")

    # Inisialisasi riwayat pesan di session state
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {
                "role": "assistant",
                "content": (
                    "Halo! Saya adalah **Rice Leaf Vision AI**, asisten pakar agronomi dan kecerdasan buatan untuk tanaman padi. "
                    "Ada yang bisa saya bantu terkait penyakit padi, anjuran pupuk, atau sistem deteksi AI ini?"
                ),
            }
        ]

    # Tombol Pintas Pertanyaan Cepat
    st.markdown('<div style="font-size: .8rem; font-weight: 700; color: #566961; margin-bottom: .4rem;">PILIHAN PERTANYAAN CEPAT:</div>', unsafe_allow_html=True)
    c_q1, c_q2, c_q3, c_q4 = st.columns(4)
    quick_prompt = None
    with c_q1:
        if st.button("🌱 Cara Penanganan", use_container_width=True):
            quick_prompt = "Bagaimana cara penanganan dan pengendalian penyakit ini?"
    with c_q2:
        if st.button("🧪 Pupuk Berimbang", use_container_width=True):
            quick_prompt = "Berapa anjuran takaran pupuk N-P-K dan silika yang tepat?"
    with c_q3:
        if st.button("🌾 Varietas Tahan", use_container_width=True):
            quick_prompt = "Varietas padi apa yang tahan terhadap penyakit ini?"
    with c_q4:
        if st.button("🔬 Analisis AI Model", use_container_width=True):
            quick_prompt = "Jelaskan alasan ilmiah model AI dan akurasinya!"

    # Render Riwayat Chat
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input Chat Form
    with st.form("chat_input_form", clear_on_submit=True):
        c_inp, c_sub = st.columns([5, 1])
        with c_inp:
            chat_input_val = st.text_input(
                "Pertanyaan:",
                placeholder="Ketik pertanyaan seputar penyakit padi, pupuk, varietas tahan, atau arsitektur AI...",
                label_visibility="collapsed",
            )
        with c_sub:
            submitted = st.form_submit_button("Kirim 📨", use_container_width=True)

    prompt_to_process = quick_prompt or (chat_input_val.strip() if submitted and chat_input_val.strip() else None)

    if prompt_to_process:
        st.session_state["chat_messages"].append({"role": "user", "content": prompt_to_process})
        with st.chat_message("user"):
            st.markdown(prompt_to_process)

        with st.chat_message("assistant"):
            with st.spinner("Asisten AI sedang menyusun jawaban..."):
                reply = get_chat_response(
                    query=prompt_to_process,
                    current_predicted_disease=active_disease,
                )
                st.markdown(reply)
                st.session_state["chat_messages"].append({"role": "assistant", "content": reply})
