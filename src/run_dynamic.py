import os
import sys
from pathlib import Path
from dotenv import load_dotenv 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.controller_agent.controller_agent import ControllerAgent
from src.utils.logger import Logger
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv(PROJECT_ROOT / ".env")

def run():
    logger = Logger()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.3,
    )

    controller = ControllerAgent(logger, llm)

    controller.plan("""
Build an AutoML pipeline for a classification dataset.

Dataset contains mixed numeric and categorical features with missing values.

Goal:
- Predict target column
- Handle missing values
- Encode categorical features
- Train multiple models
- Select best model based on accuracy
""")


if __name__ == "__main__":
    run()