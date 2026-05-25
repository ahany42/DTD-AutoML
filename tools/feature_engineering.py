from langchain_core.tools import tool

@tool
def feature_engineering(task,tool_input,prompt,data_path,llm):
    """Perform feature engineering on the dataset to create new features or transform existing ones."""
    print("=========================================================================")
    print(f"[TOOL] Feature engineering: {task}")
    print(f"[TOOL_INPUT] {tool_input}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    return "features" , "updated_data_path"