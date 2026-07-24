from agents.tools import (
    RecruitmentAgent,
    SkillGapAgent,
    LearningRecommendationAgent,
    PerformanceAgent,
    CareerRecommendationAgent
)

class AgentOrchestrator:
    def __init__(self):
        self._agents = None

    @property
    def agents(self):
        if self._agents is None:
            self._agents = {
                "recruitment": RecruitmentAgent(),
                "skill_gap": SkillGapAgent(),
                "learning": LearningRecommendationAgent(),
                "performance": PerformanceAgent(),
                "career": CareerRecommendationAgent()
            }
        return self._agents

    def execute(self, agent_choice: str, query: str) -> str:
        """
        Orchestrates the query to the specific agent.
        """
        if agent_choice in self.agents:
            agent = self.agents[agent_choice]
            return agent.run(query)

        if agent_choice == "career_path_full":
            perf = self.agents["performance"].run(query)
            gap = self.agents["skill_gap"].run(query)
            learn = self.agents["learning"].run(gap)
            career = self.agents["career"].run(query)
            return f"{perf}\n\n{gap}\n\n{learn}\n\n{career}"

        return "Agent tidak ditemukan."

orchestrator = AgentOrchestrator()
