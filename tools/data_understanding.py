from langchain_core.tools import tool

@tool
def data_understanding(task,tool_input,prompt,data_path):
    """Perform data understanding by analyzing the dataset to extract insights, identify patterns, and summarize key characteristics."""
    print("=========================================================================")
    print(f"[TOOL] Understanding data: {task}")
    print(f"[TOOL] Prompt: {prompt}")
    print(f"[TOOL] Data path: {data_path}")
    print(f"[TOOL] Tool input: {tool_input}")
    return "data_summary","new_data_summary_path"