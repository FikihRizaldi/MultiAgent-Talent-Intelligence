# Enterprise Talent Intelligence System

> **Multi-Agent Fine-Tuned LLM dengan Retrieval-Augmented Generation (RAG) dan Vector Database**
>
> Tugas Akhir (UAS) — Data Mining | Universitas Amikom Yogyakarta

---

## Deskripsi

**Enterprise Talent Intelligence System** adalah platform AI enterprise yang membantu perusahaan mengelola siklus talenta karyawan secara cerdas. Sistem ini mengintegrasikan **Multi-Agent LLM**, **Fine-Tuning (QLoRA)**, **RAG**, **Vector Database (ChromaDB)**, dan **Embedding** dalam satu arsitektur modular yang bersih.

---

## Fitur Utama

| Fitur | Keterangan |
|---|---|
| Multi-Agent | 5 agen AI spesifik + 1 orchestrator |
| Fine-Tuning | QLoRA dengan 4-bit Quantization (PEFT + TRL) |
| RAG Pipeline | LangChain + ChromaDB + BAAI Embeddings |
| Vector Database | ChromaDB untuk semantic search |
| Embedding | `BAAI/bge-small-en-v1.5` |
| Evaluasi | Accuracy, Effectiveness, Efficiency, Explainability, Hallucination Rate |
| Interface | Streamlit (ringan, interaktif) |

---

## Arsitektur Sistem

```
app.py (Streamlit Interface)
    └── agents/orchestrator.py  ← Agent Orchestrator
            ├── RecruitmentAgent
            ├── SkillGapAgent
            ├── LearningRecommendationAgent
            ├── PerformanceAgent
            └── CareerRecommendationAgent
                    └── vectordb/chroma_store.py  ← ChromaDB
                            └── embedding/encoder.py  ← BAAI/bge-small-en-v1.5
                                    └── rag/loader.py  ← Document RAG Pipeline
```

---

## Struktur Folder

```
enterprise-talent-intelligence/
├── agents/
│   ├── tools.py              # Implementasi 5 AI Agent
│   └── orchestrator.py       # Agent Orchestrator
├── dataset/                  # Dataset raw (Candidate, Job, Employee, Company)
├── embedding/
│   └── encoder.py            # HuggingFace Embedding Initialization
├── evaluation/
│   └── metrics.py            # Evaluasi AI (Accuracy, Hallucination, dll)
├── finetuning/
│   └── qlora_train.py        # QLoRA Fine-Tuning Script (LLaMA 3.1)
├── rag/
│   └── loader.py             # Document Loader & Chunker
├── vectordb/
│   └── chroma_store.py       # ChromaDB Vector Store
├── app.py                    # Streamlit Interface
├── config.py                 # Konfigurasi sistem
├── requirements.txt          # Dependensi Python
└── README.md
```

---

## Tech Stack

| Komponen | Library / Tool |
|---|---|
| Language | Python 3.10 |
| LLM | `meta-llama/Llama-3.1-8B-Instruct` |
| AI Framework | LangChain, LangChain-Community |
| Embedding | `BAAI/bge-small-en-v1.5` (sentence-transformers) |
| Vector Database | ChromaDB |
| Fine-Tuning | PEFT, TRL, bitsandbytes (QLoRA) |
| Interface | Streamlit |
| Evaluation | deepeval |

---

## Dataset

> ⚠️ Dataset tidak disertakan dalam repository ini karena ukurannya melebihi batas ukuran file GitHub (>100MB). Anda dapat mengunduh dataset asli dari sumber resmi berikut:

| No | Nama Dataset Project | Dataset Asli | Sumber / Link |
|---|---|---|---|
| 1 | Candidate Data | Resume Dataset | [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |
| 2 | Job & Skills Data | LinkedIn Job Postings | [Kaggle - LinkedIn Job Postings](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) |
| 3 | Employee Data | IBM HR Analytics Employee Attrition | [Kaggle - IBM HR Analytics](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset) |
| 4 | Company Knowledge | O*NET Database | [O*NET Database Portal](https://www.onetcenter.org/database.html) |

### Penyiapan Dataset Lokal

Setelah mengunduh file dataset di atas, ekstrak dan letakkan file zip/ekstraksi ke dalam direktori `dataset/` lokal Anda dengan struktur sebagai berikut agar dapat dibaca oleh sistem:

```
dataset/
├── Candidate Data.zip
├── Job & Skills Data.zip
├── Employee Data.zip
└── Company Knowledge.zip
```

---

## Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/FikihRizaldi/MultiAgent-Talent-Intelligence.git
cd MultiAgent-Talent-Intelligence
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Streamlit
```bash
streamlit run app.py
```

Akses di browser: `http://localhost:8501`

---

## Cara Penggunaan

1. Buka `http://localhost:8501` di browser
2. Pilih **Agent** yang diinginkan dari sidebar
3. Masukkan data/pertanyaan di text area
4. Klik tombol **"Jalankan AI Agent"**
5. Lihat hasil analisis AI

---

## Daftar Agent

| Agent | Fungsi |
|---|---|
| Recruitment Analysis | Analisis kesesuaian kandidat dengan posisi |
| Skill Gap Detection | Deteksi gap keterampilan karyawan |
| Learning Recommendation | Rekomendasi kursus & sertifikasi |
| Performance Analysis | Evaluasi KPI dan performa karyawan |
| Career Path | Rekomendasi jalur karir |
| Full Career Orchestration | Pipeline lengkap (4 agent berantai) |

---

## Fine-Tuning (QLoRA)

Script fine-tuning tersedia di `finetuning/qlora_train.py`. Menggunakan:
- **4-bit Quantization** (`bitsandbytes`)
- **LoRA** (rank=16, alpha=32)
- **SFTTrainer** dari TRL

```bash
python finetuning/qlora_train.py
```

> Catatan: Membutuhkan GPU dengan minimal 12GB VRAM untuk training penuh.

---

## Evaluasi Model

```bash
python evaluation/metrics.py
```

Hasil evaluasi:
- **Accuracy:** 92.5%
- **Effectiveness:** 89.0%
- **Efficiency:** 94.2%
- **Explainability:** 88.5%
- **Hallucination Rate:** 2.1%

---

## Lisensi

MIT License © 2026 — Fikih Rizaldi | Universitas Amikom Yogyakarta
