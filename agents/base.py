from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseAgent(ABC):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process the input context and return the agent's response/action.
        """
        pass
    
    def log_action(self, action: str, input_data: Dict, output_data: Dict, status: str = "Success"):
        # In a real app, this would write to the AgentLog database table via a session or async task
        print(f"[{self.name}] {action} - Status: {status}")
