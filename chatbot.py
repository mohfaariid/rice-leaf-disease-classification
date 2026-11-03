"""Modul Asisten Pakar Padi & AI Q&A Engine (Hybrid).

Mendukung 2 mode:
1. Local Agronomic Knowledge Engine: Bekerja 100% offline tanpa perlu API key eksternal.
2. Google Gemini REST API: Bekerja otomatis jika API key tersedia.
"""

from pathlib import Path
import os
import re
import json
import urllib.request
import urllib.error


# ==============================================================================
# BASIS PENGETAHUAN LOKAL (KNOWLEDGE BASE AGRONOMI & RISET SKRIPSI)
# ==============================================================================
KNOWLEDGE_BASE = {
    "Bacterial Leaf Blight": {
        "nama_id": "Hawar Daun Bakteri (Kresek)",
        "patogen": "Bakteri Xanthomonas oryzae pv. oryzae",
        "gejala": (
            "Bercak basah memanjang berwarna hijau keabu-abuan atau kekuningan di sepanjang tepi daun. "
            "Lesi dapat menjalar hingga daun mengering menyerupai jerami (fase kresek). Pada pagi hari yang lembap, "
            "terkadang tampak embun bakteri berwarna kuning keruh di permukaan daun."
        ),
        "penanganan": (
            "1. Kurangi pemupukan Nitrogen (Urea) berlebih yang membuat dinding sel daun sukulen/lunak.\n"
            "2. Tingkatkan pupuk Kalium (KCl) dan Silika (SiO2) untuk memperkuat dinding sel daun.\n"
            "3. Atur pengairan dengan sistem intermitten (berselang); hindari genangan air yang terlalu dalam.\n"
            "4. Bersihkan gulma inang di pematang sawah dan buang jerami yang terinfeksi parah.\n"
            "5. Gunakan bakterisida berbahan aktif tembaga (Cupper) atau asam oksolinik jika serangan mencapai ambang batas ekonomi."
        ),
        "varietas": "Inpari 32 HDB, Inpari 42 Agritan GSR, Ciherang, dan Conde.",
    },
    "Brown Spot": {
        "nama_id": "Bercak Cokelat",
        "patogen": "Jamur Bipolaris oryzae (Cochliobolus miyabeanus)",
        "gejala": (
            "Bercak oval atau bulat berwarna cokelat tua di helai daun dengan pusat bercak berwarna abu-abu keputihan "
            "atau kuning pucat. Penyakit ini sering menjadi indikator tanah yang miskin hara atau kekurangan air."
        ),
        "penanganan": (
            "1. Perbaiki kesuburan tanah dengan pupuk organik/kompos dan pemupukan berimbang.\n"
            "2. Tambahkan pupuk Kalium (K) dan pupuk mikro untuk meningkatkan daya tahan tanaman.\n"
            "3. Pastikan kecukupan air sawah; hindari fase kekeringan (water stress) pada fase vegetatif.\n"
            "4. Lakukan perlakuan benih (seed treatment) sebelum semai menggunakan air hangat atau agens hayati Trichoderma sp.\n"
            "5. Aplikasi fungisida berbahan aktif mankozeb, difenokonazol, atau azoksistrobin jika lesi meluas."
        ),
        "varietas": "Inpari 30, Inpari 33, dan varietas lokal unggul bersertifikat.",
    },
    "Healthy Rice Leaf": {
        "nama_id": "Daun Padi Sehat",
        "patogen": "Tidak ada (Fisiologis Normal)",
        "gejala": "Warna helai daun hijau merata, segar, tekstur daun kokoh tanpa bercak lesi patogenik aktif.",
        "penanganan": (
            "1. Lanjutkan pemantauan rutin mingguan (scouting) untuk mendeteksi dini jika muncul gejala baru.\n"
            "2. Pertahankan pemupukan berimbang N-P-K sesuai rekomendasi kalender tanam setempat.\n"
            "3. Jaga kebersihan pematang sawah dan sirkulasi air yang lancar.\n"
            "4. Tetap amati bagian pelepah bawah yang dekat permukaan air."
        ),
        "varietas": "Kondisi tanaman sehat dapat dipertahankan dengan benih bersertifikat dan Good Agricultural Practices (GAP).",
    },
    "Leaf Blast": {
        "nama_id": "Blas Daun (Pyricularia)",
        "patogen": "Jamur Magnaporthe oryzae (Pyricularia oryzae)",
        "gejala": (
            "Bercak khas berbentuk belah ketupat atau kumparan (spindle-shaped) dengan kedua ujung meruncing. "
            "Pusat bercak berwarna abu-abu keputihan dengan tepi cokelat kemerahan atau cokelat gelap."
        ),
        "penanganan": (
            "1. Hindari pemupukan Nitrogen dosis tinggi; seimbangkan dengan pupuk Fosfat dan Kalium.\n"
            "2. Terapkan pola tanam jajar legowo (2:1 atau 4:1) untuk memperbaiki aerasi dan mengurangi kelembapan mikro.\n"
            "3. Hindari penggunaan benih dari petak sawah yang sebelumnya pernah terserang blas.\n"
            "4. Jika bercak kumparan mulai menyebar di daun muda, semprotkan fungisida berbahan aktif trisiklazol, isoprotiolan, atau difenokonazol."
        ),
        "varietas": "Inpari 32 HDB, Inpari 48 Blas, Ciherang, dan Mekongga.",
    },
    "Leaf Scald": {
        "nama_id": "Hawar Bergaris / Scald",
        "patogen": "Jamur Microdochium albescens (Monographella albescens)",
        "gejala": (
            "Lesi berpita melingkar seperti terbakar (scalded) dengan garis konsentris berwarna zaitun kehijauan "
            "hingga cokelat pudar pada ujung atau tepi daun. Tepi lesi sering kali dibatasi garis cokelat tua sempit."
        ),
        "penanganan": (
            "1. Lakukan pemangkasan daun liar dan bersihkan rumput teki/gulma pematang yang menjadi inang jamur.\n"
            "2. Perbaiki jarak tanam agar sinar matahari dapat menembus rumpun bagian dalam.\n"
            "3. Kurangi kelembapan kanopi dengan pengeringan sawah secara periodik.\n"
            "4. Aplikasikan fungisida protektif berbahan aktif tembaga hidroksida atau mankozeb pada fase awal infeksi."
        ),
        "varietas": "Gunakan varietas berdaya adaptasi tinggi seperti Inpari 30 Ciherang Sub-1.",
    },
    "Sheath Blight": {
        "nama_id": "Hawar Pelepah",
        "patogen": "Jamur Rhizoctonia solani",
        "gejala": (
            "Bercak oval atau tidak beraturan berwarna kehijauan atau abu-abu pucat dengan tepi cokelat gelap. "
            "Gejala umumnya dimulai dari pelepah daun bawah di dekat permukaan air dan menjalar ke atas menuju helai daun."
        ),
        "penanganan": (
            "1. Turunkan ketinggian genangan air sawah (sistem pengairan basah-kering berselang/AWD).\n"
            "2. Hindari kerapatan populasi rumpun yang terlalu padat (gunakan jarak tanam minimal 25x25 cm atau jajar legowo).\n"
            "3. Bersihkan sklerosia jamur yang mengapung di permukaan air saat pengolahan tanah.\n"
            "4. Semprot fungisida berbahan aktif heksakonazol, validamisin, atau azoksistrobin diarahkan ke pangkal pelepah tanaman."
        ),
        "varietas": "Varietas berbatang kokoh dan anakan sedang lebih toleran terhadap hawar pelepah.",
    },
    "Tungro": {
        "nama_id": "Penyakit Tungro (Virus Kuning)",
        "patogen": "Infeksi ganda virus RTBV (Rice Tungro Bacilliform) dan RTSV (Rice Tungro Spherical)",
        "gejala": (
            "Daun berubah warna menjadi kuning hingga jingga cerah (orange-yellowing) mulai dari ujung helai daun. "
            "Tanaman kerdil (stunted), pertumbuhan terhambat, anakan sedikit, dan daun muda terkadang berbercak karat."
        ),
        "penanganan": (
            "1. Penyakit ini TIDAK BISA disembuhkan dengan fungisida/bakterisida karena disebabkan oleh virus.\n"
            "2. Pengendalian mutlak diarahkan pada VEKTOR PENULARNYA, yaitu Wereng Daun Hijau (Nephotettix virescens).\n"
            "3. Segera cabut dan musnahkan (eradikasi) rumpun yang terserang agar virus tidak ditularkan ke rumpun sehat.\n"
            "4. Lakukan tanam serempak dalam satu hamparan untuk memutus siklus hidup wereng hijau.\n"
            "5. Kendalikan populasi wereng hijau menggunakan insektisida imidakloprid, tiametoksam, atau buprofezin sesuai anjuran PPL."
        ),
        "varietas": "Inpari 36 Lanrang, Inpari 37 Pupuk, Tukad Balian, dan Kalimas (varietas tahan wereng hijau).",
    },
}

THESIS_FACTS = {
    "penulis": "Mohammad Farid Anshori (NIM: 22523124)",
    "kampus": "Program Studi Informatika, Universitas Islam Indonesia (UII), 2026",
    "pembimbing": "Arrie Kurniawardhani, S.Si., M.Kom.",
    "model_terbaik": "MobileNetV2 Fine-tuning 50 layer pada dataset Balanced Quality-Hybrid",
    "akurasi_terbaik": "92,26%",
    "f1_macro_terbaik": "92,17%",
    "recall_macro_terbaik": "92,21%",
    "cnn_akurasi": "76,34% (pada dataset balanced) dan 86,98% (pada dataset original)",
    "ukuran_model": "MobileNetV2: 24,1 MB (2,4 juta parameter) vs CNN Kustom: 101 MB (25,2 juta parameter)",
    "dataset": "Rice_Leaf_AUG (total 5.137 citra awal, diseimbangkan menjadi 4.571 citra dengan 653 citra per kelas)",
}


# ==============================================================================
# FUNGSI CHATBOT HYBRID
# ==============================================================================
def ask_gemini_api(prompt: str, context: str, api_key: str) -> str | None:
    """Mengirim prompt ke Google Gemini REST API tanpa dependensi protobuf."""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        
        system_instructions = (
            "Anda adalah 'Rice Leaf Vision AI', asisten pakar agronomi tanaman padi dan pakar deep learning "
            "yang dikembangkan berdasarkan penelitian skripsi Mohammad Farid Anshori (Informatika UII 2026). "
            "Tugas Anda adalah membantu petani, penyuluh, dan penguji memahami gejala penyakit daun padi, "
            "langkah penanganan ilmiah, takaran pupuk berimbang, cara pencegahan, serta aspek teknis model AI (MobileNetV2 vs CNN Kustom). "
            "Jawablah dengan ramah, profesional, praktis, dan menggunakan Bahasa Indonesia yang mudah dipahami."
        )

        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": f"SISTEM KONTEKS: {system_instructions}\nKONTEKS DIAGNOSIS DAUN SAAT INI: {context}\n\nPERTANYAAN PENGGUNA: {prompt}"}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.4,
                "maxOutputTokens": 800,
            }
        }

        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=12) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None


def ask_local_knowledge_engine(query: str, active_context_disease: str | None = None) -> str:
    """Mesin penjawab lokal cerdas berbasis aturan, NLP sederhana, dan basis data agronomi."""
    q = query.lower()

    # 1. Pertanyaan tentang Riset / Skripsi / Model AI
    if any(k in q for k in ["skripsi", "penelitian", "siapa", "penulis", "nim", "pembimbing", "uii"]):
        return (
            f"Penelitian ini dilakukan oleh **{THESIS_FACTS['penulis']}** di bawah bimbingan **{THESIS_FACTS['pembimbing']}** "
            f"pada **{THESIS_FACTS['kampus']}**.\n\n"
            f"Fokus penelitian adalah *Analisis Perbandingan Performa CNN Kustom dan MobileNetV2 pada Klasifikasi Penyakit Daun Padi Menggunakan Dataset Tidak Seimbang dan Seimbang*."
        )

    if any(k in q for k in ["model terbaik", "akurasi", "f1", "performa", "hasil", "tertinggi", "mobilenet"]):
        return (
            f"Berdasarkan hasil eksperimen Tugas Akhir:\n\n"
            f"- **Model Terbaik:** {THESIS_FACTS['model_terbaik']}.\n"
            f"- **Akurasi Pengujian:** **{THESIS_FACTS['akurasi_terbaik']}**\n"
            f"- **Macro F1-Score:** **{THESIS_FACTS['f1_macro_terbaik']}**\n"
            f"- **Macro Recall:** **{THESIS_FACTS['recall_macro_terbaik']}**\n\n"
            f"MobileNetV2 unggul telak karena memanfaatkan bobot awal (*transfer learning*) dan mekanisme *depthwise separable convolution*, "
            f"menghasilkan model yang sangat ringkas (**{THESIS_FACTS['ukuran_model']}**)."
        )

    if any(k in q for k in ["grad-cam", "gradcam", "xai", "explainable", "heatmap", "peta panas"]):
        return (
            "**Grad-CAM (Gradient-weighted Class Activation Mapping)** adalah metode *Explainable AI (XAI)* yang digunakan pada aplikasi ini. "
            "Grad-CAM menghitung gradien skor prediksi terhadap peta fitur lapisan konvolusi terakhir (`out_relu` pada MobileNetV2).\n\n"
            "Visualisasi warna merah/kuning menunjukkan area piksel bercak/lesi daun yang paling diperhatikan AI saat menentukan kelas penyakit, "
            "memastikan bahwa model mengambil keputusan berdasarkan gejala daun dan bukan karena latar belakang tanah/daun lain."
        )

    if any(k in q for k in ["imbalance", "tidak seimbang", "quality-hybrid", "balancing", "seimbang", "tungro 1308"]):
        return (
            "Pada dataset awal, kelas **Tungro berjumlah 1.308 citra**, sedangkan kelas lainnya hanya berkisar 600-an citra (*class imbalance*).\n\n"
            "Penelitian ini menerapkan strategi **Balanced Quality-Hybrid**, yaitu menyaring citra Tungro berdasarkan kualitas visual (menyisakan 653 citra terbaik) "
            "dan menambah kelas minoritas via augmentasi (+89 citra). Hasilnya, performa MobileNetV2 melonjak dari 87,89% menjadi **92,26%**."
        )

    # 2. Identifikasi Kelas Penyakit dari Pertanyaan atau Konteks Aktif
    target_class = active_context_disease
    for class_key in KNOWLEDGE_BASE.keys():
        if class_key.lower() in q or KNOWLEDGE_BASE[class_key]["nama_id"].lower() in q:
            target_class = class_key
            break

    # Alias pencarian bahasa indonesia
    if "kresek" in q or "bakteri" in q or "blight" in q:
        target_class = "Bacterial Leaf Blight"
    elif "bercak cokelat" in q or "brown spot" in q:
        target_class = "Brown Spot"
    elif "sehat" in q or "healthy" in q:
        target_class = "Healthy Rice Leaf"
    elif "blas" in q or "blast" in q or "pyricularia" in q:
        target_class = "Leaf Blast"
    elif "scald" in q or "garis" in q:
        target_class = "Leaf Scald"
    elif "pelepah" in q or "sheath" in q:
        target_class = "Sheath Blight"
    elif "tungro" in q or "kuning" in q or "wereng" in q:
        target_class = "Tungro"

    if target_class and target_class in KNOWLEDGE_BASE:
        info = KNOWLEDGE_BASE[target_class]
        
        # Pertanyaan tentang obat / penanganan / langkah
        if any(k in q for k in ["obat", "atasi", "penanganan", "cegah", "solusi", "fungisida", "bakterisida", "cara"]):
            return (
                f"### 🌾 Langkah Penanganan untuk **{info['nama_id']}**:\n"
                f"**Penyebab:** {info['patogen']}\n\n"
                f"{info['penanganan']}\n\n"
                f"💡 **Rekomendasi Varietas Tahan:** {info['varietas']}"
            )
        
        # Pertanyaan tentang pupuk / nutrisi
        if any(k in q for k in ["pupuk", "urea", "nitrogen", "kalium", "phonska", "dosis", "nutrisi"]):
            return (
                f"### 🧪 Panduan Pemupukan Terkait **{info['nama_id']}**:\n"
                f"- **Kurangi Nitrogen (Urea):** Pemupukan Urea yang berlebihan membuat dinding sel tanaman menjadi sukulen dan lunak, sangat mudah ditembus oleh patogen.\n"
                f"- **Tingkatkan Kalium (KCl):** Kalium mempertebal dinding sel epidermis daun sehingga lebih tahan penetrasi jamur/bakteri.\n"
                f"- **Gunakan Silika (SiO2):** Unsur silika membentuk lapisan ganda silika-selulosa pada daun padi yang berfungsi sebagai perisai mekanis alami terhadap serangan penyakit daun."
            )

        # Pertanyaan tentang gejala / tanda
        if any(k in q for k in ["gejala", "tanda", "ciri", "kenapa", "bentuk", "penyebab"]):
            return (
                f"### 🔍 Karakteristik Visual **{info['nama_id']}**:\n\n"
                f"- **Penyebab:** {info['patogen']}\n"
                f"- **Ciri Khas:** {info['gejala']}\n\n"
                f"💡 *Jika menemukan gejala ini di sawah, segera periksa rumpun di sekitarnya untuk melihat apakah serangannya merata atau terlokalisir.*"
            )

        # Pertanyaan tentang varietas tahan
        if any(k in q for k in ["varietas", "benih", "bibit", "tahan"]):
            return (
                f"### 🌾 Rekomendasi Varietas Tahan untuk **{info['nama_id']}**:\n\n"
                f"Untuk meminimalkan risiko serangan penyakit ini pada musim tanam berikutnya, disarankan menggunakan benih bersertifikat seperti:\n"
                f"👉 **{info['varietas']}**"
            )

        # Jawaban umum tentang kelas tersebut
        return (
            f"### Informasi Mengenai **{info['nama_id']}**\n\n"
            f"- **Penyebab:** {info['patogen']}\n"
            f"- **Gejala Utama:** {info['gejala']}\n\n"
            f"**Anjuran Penanganan:**\n{info['penanganan']}\n\n"
            f"**Varietas Padi Anjuran:** {info['varietas']}"
        )

    # 3. Respon Umum Pertanian Padi jika tidak spesifik
    return (
        "Halo! Saya adalah **Asisten Pakar Padi AI (Rice Leaf Vision)**.\n\n"
        "Saya dapat membantu Anda menjawab seputar:\n"
        "1. **7 Penyakit Daun Padi:** Hawar Bakteri (Kresek), Bercak Cokelat, Daun Sehat, Blas Daun, Scald, Hawar Pelepah, dan Tungro.\n"
        "2. **Solusi & Penanganan Lapangan:** Rekomendasi pemupukan berimbang (N-P-K & Silika), pengaturan air intermitten, serta fungisida/bakterisida yang tepat.\n"
        "3. **Riset Model AI:** Informasi akurasi model MobileNetV2 (92,26%), CNN Kustom, Grad-CAM, dan dataset skripsi.\n\n"
        "💡 *Silakan tanyakan hal spesifik, misalnya: 'Bagaimana cara mengatasi penyakit blas?' atau 'Pupuk apa yang cocok untuk daun menguning?'*"
    )


def get_chat_response(query: str, current_predicted_disease: str | None = None, api_key: str | None = None) -> str:
    """Fungsi utama untuk mendapatkan jawaban chat (mencoba Gemini API terlebih dahulu, fallback ke engine lokal)."""
    # 1. Coba Gemini API jika ada key
    key = api_key or os.getenv("GEMINI_API_KEY", "").strip()
    if key:
        context_str = f"Penyakit daun padi yang baru saja dideteksi sistem adalah: {current_predicted_disease}" if current_predicted_disease else "Belum ada citra yang dideteksi."
        gemini_reply = ask_gemini_api(query, context_str, key)
        if gemini_reply:
            return gemini_reply

    # 2. Fallback ke Mesin Pengetahuan Lokal (100% Offline & Instan)
    return ask_local_knowledge_engine(query, current_predicted_disease)
