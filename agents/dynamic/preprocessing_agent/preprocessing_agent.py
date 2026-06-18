"""PreprocessingAgent LangGraph orchestrator."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# Add project root to Python path so imports work from anywhere.
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from agents.dynamic.preprocessing_agent.graph import build_preprocessing_graph
from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState
from tools.pipeline_state import empty_state


class PreprocessingAgent:
    """
    Run preprocessing and then LLM-guided feature engineering in one graph.

    The agent registers the feature-engineering tool automatically, so existing
    callers only need to keep registering preprocessing_execution.
    """

    def __init__(self, logger: Any, llm: Any, registry: Any):
        self.logger = logger
        self.llm = llm
        self.registry = registry

        if self.registry.get("feature_engineering_execution") is None:
            from tools.feature_engineering_execution import (
                feature_engineering_execution,
            )

            self.registry.register(
                "feature_engineering_execution",
                feature_engineering_execution,
            )

    def run(
        self,
        data_path: str,
        prompt: str,
        pipeline_state: dict | None = None,
        *,
        task: str = "Execute preprocessing pipeline",
        target_column: str = "",
        test_size: float = 0.2,
        use_llm: bool = True,
        preprocessing_input: dict | None = None,
        feature_top_k: int = 4,
        feature_engineering_input: dict | None = None,
    ) -> dict:
        """
        Execute preprocessing followed by feature engineering.

        The returned pipeline state includes both preprocessing_output and
        feature_engineering_output. The engineered train/test CSVs preserve all
        original preprocessed columns and append only the top 3-4 new columns.
        """
        config = {
            "target_column": target_column,
            "test_size": test_size,
            "use_llm": use_llm,
            "preprocessing_input": preprocessing_input or {},
            "feature_top_k": feature_top_k,
            "feature_engineering_input": feature_engineering_input or {},
        }

        graph = build_preprocessing_graph(self.llm, self.registry, config)
        initial: PreprocessingAgentState = {
            "data_path": data_path,
            "prompt": prompt,
            "task": task,
            "pipeline_state": pipeline_state or empty_state(data_path, prompt),
            "step": "preprocessing_agent_start",
        }

        self.logger.info("\n" + "=" * 50)
        self.logger.info("PREPROCESSING + FEATURE ENGINEERING AGENT (LangGraph)")
        self.logger.info("=" * 50)

        final_state: PreprocessingAgentState = graph.invoke(initial)
        final_pipeline_state = (
            final_state.get("pipeline_state") or initial["pipeline_state"]
        )

        if final_state.get("error"):
            self.logger.warning(
                f"PreprocessingAgent finished with error: {final_state['error']}"
            )
        else:
            self.logger.info(
                "PreprocessingAgent finished - "
                f"step={final_pipeline_state.get('step')} "
                f"status={final_pipeline_state.get('status')}"
            )

        return final_pipeline_state


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.utils.logger import Logger
    from tools.feature_engineering_execution import feature_engineering_execution
    from tools.preprocessing_execution import preprocessing_execution
    from tools.registry import ToolRegistry

    load_dotenv()

    # ======================================================================
    # CONFIGURATION BLOCK
    # ======================================================================

    DATASET_NAME = "Titanic-Dataset.csv"
    DATASET_PATH = None

    TARGET_COLUMN = "Survived"
    TEST_SIZE = 0.2
    USE_LLM = True
    FEATURE_TOP_K = 4

    LLM_MODEL = "gemini-2.5-flash"
    LLM_TEMPERATURE = 0.3
    LLM_API_KEY = os.getenv("GOOGLE_API_KEY")

    TASK_NAME = "Preprocess Titanic dataset and engineer features"
    TASK_PROMPT = (
        "Clean and preprocess the Titanic dataset for classification, then "
        "create useful feature combinations."
    )

    # ======================================================================
    # END CONFIGURATION BLOCK
    # ======================================================================

    logger = Logger()
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        google_api_key=LLM_API_KEY,
        temperature=LLM_TEMPERATURE,
    )
    registry = ToolRegistry()
    registry.register("preprocessing_execution", preprocessing_execution)
    registry.register(
        "feature_engineering_execution",
        feature_engineering_execution,
    )

    data_path = (
        DATASET_PATH
        if DATASET_PATH
        else str(_project_root / "uploads" / DATASET_NAME)
    )

    if not Path(data_path).exists():
        print(f"Dataset not found: {data_path}")
        raise SystemExit(1)

    agent = PreprocessingAgent(logger, llm, registry)
    result = agent.run(
        data_path=data_path,
        prompt=TASK_PROMPT,
        task=TASK_NAME,
        target_column=TARGET_COLUMN,
        test_size=TEST_SIZE,
        use_llm=USE_LLM,
        feature_top_k=FEATURE_TOP_K,
    )

    print("\n" + "=" * 70)
    print("PREPROCESSING + FEATURE ENGINEERING COMPLETE")
    print("=" * 70)
    print(f"Status: {result.get('status')}")
    if result.get("status") == "success":
        preprocessing_output = result.get("preprocessing_output", {})
        feature_output = result.get("feature_engineering_output", {})
        print(f"X_train:            {preprocessing_output.get('X_train_path')}")
        print(f"X_test:             {preprocessing_output.get('X_test_path')}")
        print(f"y_train:            {preprocessing_output.get('y_train_path')}")
        print(f"y_test:             {preprocessing_output.get('y_test_path')}")
        print(
            "X_train engineered: "
            f"{feature_output.get('X_train_engineered_path')}"
        )
        print(
            "X_test engineered:  "
            f"{feature_output.get('X_test_engineered_path')}"
        )
        print(f"Feature report:      {feature_output.get('feature_report_path')}")
        print(
            "Selected new cols:   "
            f"{feature_output.get('selected_features', [])}"
        )
    else:
        print(f"Error: {result.get('error')}")
    print("=" * 70 + "\n")
