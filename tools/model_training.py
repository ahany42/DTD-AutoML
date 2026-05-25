from langchain_core.tools import tool

@tool
def model_training(task,tool_input,prompt,data_path,llm):
    """Train a machine learning model based on the provided data and specifications."""
    print("=========================================================================")
    print(f"[TOOL] Training model: {task}")
    print(f"[TOOL_INPUT] {tool_input}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    return "model","new_model_path"