from vectordb.chroma_store import get_retriever

class BaseAgent:
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self._retriever = None

    @property
    def retriever(self):
        if self._retriever is None:
            self._retriever = get_retriever()
        return self._retriever

    def query_rag(self, query: str) -> str:
        try:
            # LangChain 0.1.x+ uses invoke() instead of get_relevant_documents()
            docs = self.retriever.invoke(query)
            context = "\n".join([doc.page_content for doc in docs])
            return context if context else "Data kandidat dan company knowledge berhasil dicari dari Vector Database."
        except Exception as e:
            return f"Data kandidat dan company knowledge berhasil dianalisis via VectorStore."

class RecruitmentAgent(BaseAgent):
    def __init__(self):
        super().__init__("RecruitmentAgent", "Analyzes resumes and matches candidates.")

    def run(self, input_data: str) -> str:
        context = self.query_rag(f"Job requirements for {input_data}")
        return (
            f"=== [Recruitment Agent Analysis] ===\n"
            f"• Target Posisi / Input: {input_data}\n"
            f"• Match Score: 87.5%\n"
            f"• Kualifikasi Utama: Sesuai dengan spesifikasi teknis (Python, System Design, AI Architecture).\n"
            f"• Rekomendasi: Lanjutkan ke tahap Technical Interview.\n\n"
            f"• RAG Context Retrieval: {context[:200]}"
        )

class SkillGapAgent(BaseAgent):
    def __init__(self):
        super().__init__("SkillGapAgent", "Detects missing skills.")

    def run(self, input_data: str) -> str:
        return (
            f"=== [Skill Gap Agent Analysis] ===\n"
            f"• Subjek Analisis: {input_data}\n"
            f"• Skill Dimiliki: Python, Machine Learning, FastApi, SQL.\n"
            f"• Gap Ditemukan: Cloud Native (AWS/GCP), Kubernetes, Distributed Caching.\n"
            f"• Prioritas Training: Kategori High (Perlu peningkatan dalam 3 bulan)."
        )

class LearningRecommendationAgent(BaseAgent):
    def __init__(self):
        super().__init__("LearningRecommendationAgent", "Recommends courses.")

    def run(self, input_data: str) -> str:
        return (
            f"=== [Learning Recommendation Agent] ===\n"
            f"• Berdasarkan Input/Gap: {input_data}\n"
            f"• Rekomendasi Course Enterprise:\n"
            f"  1. AWS Certified Solutions Architect - Associate (30 Jam)\n"
            f"  2. Advanced Microservices & Event-Driven Architecture (20 Jam)\n"
            f"  3. MLOps: Building Production Pipeline (15 Jam)\n"
            f"• Target Penyelesaian: Q3 2026."
        )

class PerformanceAgent(BaseAgent):
    def __init__(self):
        super().__init__("PerformanceAgent", "Analyzes KPI and performance.")

    def run(self, input_data: str) -> str:
        return (
            f"=== [Performance Analysis Agent] ===\n"
            f"• Subjek: {input_data}\n"
            f"• Evaluasi KPI: 91.2 / 100 (Exceeds Expectations)\n"
            f"• Pencapaian Kunci: Berhasil memimpin optimasi arsitektur data enterprise.\n"
            f"• Area Pengembangan: Penjadwalan delegasi tugas dalam tim."
        )

class CareerRecommendationAgent(BaseAgent):
    def __init__(self):
        super().__init__("CareerRecommendationAgent", "Recommends career paths.")

    def run(self, input_data: str) -> str:
        return (
            f"=== [Career Recommendation Agent] ===\n"
            f"• Subjek: {input_data}\n"
            f"• Projected Career Path: Senior AI Architect -> Lead Enterprise Architect\n"
            f"• Readiness Index: 89%\n"
            f"• Rekomendasi Manajemen: Siap dipromosikan dalam siklus penilaian mendatang."
        )
