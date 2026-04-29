import json
from langchain_core.messages import SystemMessage, HumanMessage
class ControllerAgent:
    def __init__(self, logger, llm):
        self.logger = logger
        self.llm =llm 
    def plan(self, prompt: str):
        print("Plan Started")
        self.logger.info("\n" + "=" * 50)
        self.logger.info("LLM AGENT PLANNING MODE")
        self.logger.info("=" * 50)

        system_prompt = """
You are an AutoML planner agent.

Your job is to convert a user request into an ordered execution plan.

Available agents:
- Data Understanding Agent
- Data Cleaning Agent
- Feature Engineering Agent
- Deterministic Training Agent
- Dynamic AutoML Agent
- Evaluation Agent

Rules:
- Always start with Data Understanding Agent
- Always end with Evaluation Agent
- Choose intermediate agents based on the task
- Return ONLY valid JSON

Output format:
{
  "plan": [
    "Agent 1",
    "Agent 2",
    ...
  ]
}
"""

        response = self.llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
        ])

        response = response.content

        try:
            result = json.loads(response)
            plan = result["plan"]
        except Exception:
            self.logger.error("Failed to parse LLM output")
            self.logger.info("Raw output:")
            self.logger.info(response)
            return []

        self.logger.info("Execution Plan:\n")

        for i, step in enumerate(plan, 1):
            self.logger.info(f"{i}. {step}")

        return plan