from langchain_core.tools import tool

@tool
def data_cleaning(task,tool_input,prompt,data_path):
    """Perform data cleaning by identifying and handling missing values, outliers, and inconsistencies in the dataset."""
    print("=========================================================================")
    print("task:", task)
    print("input_data:", tool_input)
    print("input_data:", data_path)
    print("prompt:", prompt)
    return "clean_data","new_clean_path"