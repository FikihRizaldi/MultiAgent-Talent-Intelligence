import streamlit as st
from agents.orchestrator import AgentOrchestrator

st.set_page_config(page_title="Enterprise Talent Intelligence AI", layout="wide")

st.title("Enterprise Talent Intelligence System (AI Core)")
st.write("Sistem Multi-Agent berbasis LLM dengan RAG, VectorDB (Chroma), dan Fine-Tuning (QLoRA).")

@st.cache_resource
def get_orchestrator():
    return AgentOrchestrator()

orchestrator = get_orchestrator()

# Sidebar
st.sidebar.header("Agent Selector")
agent_options = {
    "Recruitment Analysis": "recruitment",
    "Skill Gap Detection": "skill_gap",
    "Learning Recommendation": "learning",
    "Performance Analysis": "performance",
    "Career Path": "career",
    "Full Career Orchestration": "career_path_full"
}
selected_agent = st.sidebar.selectbox("Pilih Agent:", list(agent_options.keys()))

st.subheader(f"Agent Aktif: {selected_agent}")

query = st.text_area("Masukkan Instruksi / Data Candidate / Prompt Pertanyaan:", "Bantu analisis kandidat ini untuk posisi Senior Developer...")

if st.button("Jalankan AI Agent", type="primary"):
    with st.spinner("Agent sedang memproses..."):
        agent_key = agent_options[selected_agent]
        response = orchestrator.execute(agent_key, query)

    st.success("Analisis Selesai!")
    st.write("### Hasil Analisis AI Agent:")
    st.info(response)
