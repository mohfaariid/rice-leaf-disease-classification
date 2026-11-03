"""Konstanta dan konfigurasi global untuk Rice Leaf Vision."""

from pathlib import Path

# Jalur Berkas dan Direktori
PROJECT_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = PROJECT_ROOT.parent

MOBILENET_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "mobilenetv2_balanced_quality_hybrid_653_ablation_50_layers_best_val_loss.keras"
)
CUSTOM_CNN_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "custom_cnn_balanced_quality_hybrid_653_best_val_loss.keras"
)
FIELD_PHOTOS_DIR = PROJECT_ROOT / "data" / "foto_lapangan"
AUG_DATASET_DIR = WORKSPACE_ROOT / "Rice_Leaf_AUG"
STATIC_DIR = PROJECT_ROOT / "static"
CSS_FILE = STATIC_DIR / "style.css"

IMAGE_SIZE = 256

CLASS_NAMES = [
    "Bacterial Leaf Blight",
    "Brown Spot",
    "Healthy Rice Leaf",
    "Leaf Blast",
    "Leaf Scald",
    "Sheath Blight",
    "Tungro",
]

CLASS_NAMES_ID = {
    "Bacterial Leaf Blight": "Hawar Daun Bakteri (Bacterial Leaf Blight)",
    "Brown Spot": "Bercak Cokelat (Brown Spot)",
    "Healthy Rice Leaf": "Daun Padi Sehat (Healthy Rice Leaf)",
    "Leaf Blast": "Blas Daun (Leaf Blast)",
    "Leaf Scald": "Hawar Bergaris / Scald (Leaf Scald)",
    "Sheath Blight": "Hawar Pelepah (Sheath Blight)",
    "Tungro": "Penyakit Tungro (Tungro)",
}

CLASS_GUIDANCE_ID = {
    "Bacterial Leaf Blight": {
        "analysis": "Pola visual mengindikasikan gejala Hawar Daun Bakteri (Xanthomonas oryzae). Gejala khas berupa garis basah memanjang berwarna hijau keabu-abuan atau kekuningan yang berkembang di sepanjang tepi helai daun dan dapat mengering menyerupai jerami.",
        "actions": [
            "Periksa rumpun di sekitarnya dan petakan area sawah yang menunjukkan gejala serupa.",
            "Jaga sanitasi alat pertanian dan hindari memindahkan bibit/alat basah dari petak yang terinfeksi.",
            "Atur drainase sawah, hindari genangan air yang terlalu tinggi untuk memutus rute penularan bakteri.",
            "Batasi pemupukan Nitrogen (N) berlebih dan seimbangkan dengan Kalium (K).",
        ],
        "avoid": "Jangan langsung menyemprot bakterisida kimiawi sebelum memastikan luas serangan dan berkonsultasi dengan penyuluh pertanian lapangan (PPL).",
    },
    "Brown Spot": {
        "analysis": "Pola visual mengindikasikan gejala Bercak Cokelat (Bipolaris oryzae). Lesi khas berbentuk oval atau bulat berwarna cokelat gelap dengan bagian tengah terkadang berwarna abu-abu keputihan.",
        "actions": [
            "Evaluasi kesuburan tanah, penyakit ini sering muncul akibat defisiensi hara (terutama Kalium dan Silika).",
            "Bersihkan gulma dan sisa jerami tanaman yang terinfeksi setelah panen.",
            "Pastikan sistem irigasi mencukupi dan hindari kondisi tanah terlalu kering (water stress).",
        ],
        "avoid": "Hindari menganggap satu bercak sebagai diagnosa mutlak; amati beberapa rumpun sekaligus karena banyak patogen memicu bercak serupa.",
    },
    "Healthy Rice Leaf": {
        "analysis": "Citra diklasifikasikan sebagai daun padi sehat. Warna hijau merata dengan struktur helai daun normal tanpa lesi aktif patogenik.",
        "actions": [
            "Lanjutkan pemantauan (scouting) rutin setiap minggu untuk mendeteksi dini jika muncul gejala baru.",
            "Pertahankan pemupukan berimbang (N-P-K) dan pengairan teratur.",
            "Jika ada kecurigaan gejala halus di pelepah bawah, periksa juga pangkal batang tanaman.",
        ],
        "avoid": "Prediksi ini berlaku untuk permukaan daun yang difoto. Tetap pantau pelepah dan pangkal rumpun yang tidak terlihat di kamera.",
    },
    "Leaf Blast": {
        "analysis": "Pola visual mengindikasikan gejala Blas Daun (Magnaporthe oryzae). Gejala khas berupa bercak berbentuk belah ketupat atau kumparan (spindle-shaped) dengan bagian tengah abu-abu dan pinggiran cokelat kemerahan.",
        "actions": [
            "Pantau daun-daun muda di sekitarnya karena spora jamur blas dapat menyebar sangat cepat melalui angin.",
            "Kurangi pemupukan Nitrogen dosis tinggi yang membuat dinding sel daun terlalu lunak.",
            "Rencanakan penggunaan varietas tahan blas pada musim tanam berikutnya.",
        ],
        "avoid": "Jangan tunda tindakan pengendalian jika lesi kumparan meluas di petakan, segera koordinasikan fungisida anjuran PPL.",
    },
    "Leaf Scald": {
        "analysis": "Pola visual mengindikasikan gejala Hawar Bergaris / Scald (Microdochium albescens). Ditandai dengan lesi berpita konsentris berwarna cokelat terang hingga pudar seperti terbakar pada ujung atau tepi daun.",
        "actions": [
            "Periksa sirkulasi udara pada rumpun padi yang terlalu rapat.",
            "Bersihkan gulma pematang sawah yang dapat menjadi inang alternatif jamur.",
            "Gunakan pola tanam seperti Jajar Legowo untuk menjaga aerasi dan kelembapan mikro.",
        ],
        "avoid": "Hindari diagnosis tergesa-gesa karena gejala scald kerap menyerupai daun terbakar akibat herbisida atau cuaca panas.",
    },
    "Sheath Blight": {
        "analysis": "Pola visual mengindikasikan gejala Hawar Pelepah (Rhizoctonia solani). Bercak oval tidak beraturan berwarna kehijauan atau abu-abu pucat dengan tepi cokelat tua, sering bermula dari pelepah dekat permukaan air.",
        "actions": [
            "Periksa pangkal pelepah di dekat muka air, bukan hanya permukaan daun atas.",
            "Lakukan pengeringan berkala (intermittent irrigation) untuk menurunkan kelembapan kanopi mikro.",
            "Hindari penggunaan bibit dengan kerapatan terlalu tinggi.",
        ],
        "avoid": "Jangan hanya memeriksa daun atas karena serangan utama jamur ini bermula di pangkal pelepah bawah.",
    },
    "Tungro": {
        "analysis": "Pola visual mengindikasikan gejala Penyakit Tungro (kombinasi virus RTBV dan RTSV). Gejala berupa perubahan warna daun menjadi kuning kejinggaan dari ujung helai, tanaman mengerdil, dan jumlah anakan berkurang.",
        "actions": [
            "Periksa keberadaan populasi wereng hijau (Nephotettix virescens) sebagai vektor pembawa virus.",
            "Amati apakah gejala muncul berkelompok (patchy) di petakan sawah.",
            "Laporkan segera ke petugas Pengamat Organisme Pengganggu Tumbuhan (POPT) / dinas pertanian setempat.",
        ],
        "avoid": "Jangan menyemprot fungisida/bakterisida karena tungro disebabkan virus; fokus tindakan adalah pengendalian wereng hijau dan eradikasi tanaman sakit.",
    },
}
