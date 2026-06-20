"""
tools/prompt_builder.py
D.T.D (Data To Deployment) — Multi-Agent AutoML Pipeline

Tool: Prompt Builder
Responsibility:
    Central store of all agent system-prompt templates.
    Each agent calls its build_prompt_*() function and receives
    a fully formatted string ready to send to the LLM.

    Keeping prompts here means:
        - Prompt edits never require touching agent logic
        - Every agent's prompt is readable in one place
        - Templates are unit-testable without instantiating agents

Current coverage: Agent 0 (Intent Detector).
Stubs for Agents 1-7 are included — fill each when that agent is built.
"""


# ─────────────────────────────────────────────
# Agent 0 — Intent Detector & Router
# ─────────────────────────────────────────────

_INTENT_DETECTOR_TEMPLATE = """\
You are an intent detection system for the D.T.D AutoML pipeline.

You will receive:
- User request: {nl_query}
- Dataset schema: columns={columns}, dtypes={dtypes}, shape={shape}

Available pipeline steps:
1. EDA — dataset profiling, statistical summaries, visualizations
2. Preprocessing — cleaning, encoding, scaling, class imbalance correction
3. Feature Engineering — feature synthesis, selection, dimensionality reduction
4. Model Selection — choose optimal training backend and model families
5. Model Training — train with AutoGluon / XGBoost / sklearn + Optuna HPO
6. Evaluation — metrics, SHAP explanations, self-debugging diagnostics
7. Deployment — FastAPI prediction endpoint + Dockerfile

Your task: determine which steps are needed based on the user's request.
Rules:
- If the user says "just preprocess" or "only clean my data" → only run_preprocessing=true
- If the user says "train a model" → run_preprocessing, run_model_selection, run_training, run_evaluation = true
- If the user says "full pipeline" or "end to end" → all flags = true
- If target column is not mentioned, infer it from the dataset schema or set to null
- If task type cannot be determined, set to "unknown"

Output ONLY a valid JSON object matching this schema — no explanation, no markdown:
{{
  "run_eda": true/false,
  "run_preprocessing": true/false,
  "run_feature_engineering": true/false,
  "run_model_selection": true/false,
  "run_training": true/false,
  "run_evaluation": true/false,
  "run_deployment": true/false,
  "target_column": "column_name" or null,
  "task_type": "classification" | "regression" | "clustering" | "unknown"
}}
"""


def build_prompt_intent_detector(
    nl_query: str,
    columns: list,
    dtypes: dict,
    shape: tuple,
) -> str:
    """
    Build the system prompt for Agent 0.

    Args:
        nl_query: User's NL request.
        columns:  List of column names from schema extraction.
        dtypes:   {"col": "dtype"} dict from schema extraction.
        shape:    (n_rows, n_cols) tuple.

    Returns:
        Fully formatted prompt string.
    """
    return _INTENT_DETECTOR_TEMPLATE.format(
        nl_query=nl_query,
        columns=columns,
        dtypes=dtypes,
        shape=shape,
    )


# ─────────────────────────────────────────────
# Stubs for Agents 1-7 (fill when each agent is built)
# ─────────────────────────────────────────────

def build_prompt_eda(**kwargs) -> str:
    raise NotImplementedError("EDA prompt not yet implemented")

def build_prompt_preprocessing(**kwargs) -> str:
    raise NotImplementedError("Preprocessing prompt not yet implemented")

def build_prompt_feature_engineering(**kwargs) -> str:
    raise NotImplementedError("Feature Engineering prompt not yet implemented")

def build_prompt_model_selection(**kwargs) -> str:
    raise NotImplementedError("Model Selection prompt not yet implemented")

def build_prompt_training(**kwargs) -> str:
    raise NotImplementedError("Training prompt not yet implemented")

def build_prompt_evaluation(**kwargs) -> str:
    raise NotImplementedError("Evaluation prompt not yet implemented")

def build_prompt_deployment(**kwargs) -> str:
    raise NotImplementedError("Deployment prompt not yet implemented")