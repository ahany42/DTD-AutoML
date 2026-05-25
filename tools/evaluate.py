from langchain_core.tools import tool

@tool
def evaluate(task,tool_input,prompt,data_path,llm):
    """Evaluate the performance of a trained machine learning model using appropriate metrics and techniques."""
    print("=========================================================================")
    print(f"[TOOL] Evaluating model: {task}")
    print(f"[TOOL_INPUT] {tool_input}")
    print(f"[PROMPT] {prompt}")
    print(f"[DATA_PATH] {data_path}")
    return "accuracy=0.92" , "evaluation_report_path"