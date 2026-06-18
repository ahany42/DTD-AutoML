"""LangGraph workflow for preprocessing followed by feature engineering."""
from __future__ import annotations

from typing import Any, Literal

from langgraph.graph import END, StateGraph

from agents.dynamic.preprocessing_agent.nodes.feature_engineering import (
    make_feature_engineering_node,
)
from agents.dynamic.preprocessing_agent.nodes.preprocessing import make_preprocessing_node
from agents.dynamic.preprocessing_agent.state import PreprocessingAgentState


def _route_after_preprocessing(
    state: PreprocessingAgentState,
) -> Literal["feature_engineering", "end"]:
    """Do not attempt feature engineering when preprocessing failed."""
    if state.get("error"):
        return "end"
    pipeline_state = state.get("pipeline_state") or {}
    if pipeline_state.get("status") != "success":
        return "end"
    return "feature_engineering"


def build_preprocessing_graph(llm: Any, registry: Any, config: dict | None = None):
    """
    Compile the dynamic preprocessing workflow.

    preprocessing -> feature_engineering -> END
    """
    cfg = dict(config or {})

    workflow = StateGraph(PreprocessingAgentState)
    workflow.add_node(
        "preprocessing",
        make_preprocessing_node(llm, registry, cfg),
    )
    workflow.add_node(
        "feature_engineering",
        make_feature_engineering_node(llm, registry, cfg),
    )

    workflow.set_entry_point("preprocessing")
    workflow.add_conditional_edges(
        "preprocessing",
        _route_after_preprocessing,
        {
            "feature_engineering": "feature_engineering",
            "end": END,
        },
    )
    workflow.add_edge("feature_engineering", END)

    return workflow.compile()
