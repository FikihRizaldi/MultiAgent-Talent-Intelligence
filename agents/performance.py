from typing import Any, Dict
from agents.base import BaseAgent

class PerformanceAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PerformanceAgent",
            description="Analyzes KPIs, employee performance, and generates summaries."
        )
        
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        task = context.get("task")
        
        result = {}
        if task == "analyze_performance":
            result = {"kpi_score": 4.5, "summary": "Excellent performance, exceeded goals."}
            
        return result
