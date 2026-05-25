import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_core.tools import tool


@tool
def data_understanding(task, tool_input, prompt, data_path, llm):
    """
    Perform intelligent exploratory data analysis
    and return ONLY structured JSON results.
    """

    print("=========================================================================")

    try:
        df = pd.read_csv(data_path)

      
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"output/dynamic_pipeline/{timestamp}"
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        dataset_info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {
                col: str(dtype)
                for col, dtype in df.dtypes.items()
            },
            "missing_values": df.isnull().sum().to_dict(),
            "sample_rows": df.head(3).to_dict(orient="records")
        }

        eda_prompt = f"""
        You are a senior data scientist.

        Analyze the dataset and return ONLY valid JSON.

        IMPORTANT RULES:
        - Return ONLY JSON
        - No markdown
        - No code blocks
        - No explanations outside JSON
        - No HTML

        JSON schema:

        {{
        "title": "string",

        "summary": "string",

        "sections": [
            {{
            "title": "string",

            "content": [
                {{
                "type": "text|bullet|warning|metric",

                "label": "optional",

                "value": "string"
                }}
            ]
            }}
        ],

        "visualizations": [
            {{
            "plot_type": "histogram|boxplot|scatterplot|heatmap|countplot|missing_values",
            "columns": ["column1", "column2"],
            "title": "string",
            "reason": "string"
            }}
        ],

        "recommendations": [
            "string"
        ]
        }}

        Dataset metadata:
        {json.dumps(dataset_info, indent=2)}

        Task:
        {task}

        Additional Prompt:
        {prompt}

        Tool Input:
        {tool_input}
        """

        result = llm.invoke(eda_prompt)

        response_text = getattr(result, "content", str(result))
        response_text = (
            response_text
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        report_json = json.loads(response_text)
        generated_plots = []

        for idx, viz in enumerate(
            report_json.get("visualizations", [])
        ):

            try:

                plot_type = viz.get("plot_type")
                columns = viz.get("columns", [])
                title = viz.get("title", "Visualization")
                reason = viz.get("reason", "")
                plot_filename = f"plot_{idx}.png"
                plot_path = os.path.join(
                    plots_dir,
                    plot_filename
                )

                relative_plot_path = (
                    f"/dynamic_pipeline/{timestamp}/plots/{plot_filename}"
                )

                plt.figure(figsize=(8, 5))

                # =================================================
                # MISSING VALUES
                # =================================================
                if plot_type == "missing_values":

                    missing = df.isnull().sum()

                    missing = missing[missing > 0]

                    if len(missing) > 0:

                        missing.sort_values(
                            ascending=False
                        ).plot(kind="bar")

                # =================================================
                # HEATMAP
                # =================================================
                elif plot_type == "heatmap":

                    numeric_df = df.select_dtypes(
                        include=["number"]
                    )

                    if not numeric_df.empty:

                        sns.heatmap(
                            numeric_df.corr(),
                            annot=True,
                            cmap="coolwarm"
                        )

                # =================================================
                # HISTOGRAM
                # =================================================
                elif plot_type == "histogram":

                    if columns and columns[0] in df.columns:

                        sns.histplot(
                            df[columns[0]].dropna()
                        )

                # =================================================
                # BOXPLOT
                # =================================================
                elif plot_type == "boxplot":

                    if len(columns) == 1:

                        sns.boxplot(
                            x=df[columns[0]]
                        )

                    elif len(columns) >= 2:

                        sns.boxplot(
                            x=df[columns[0]],
                            y=df[columns[1]]
                        )

                # =================================================
                # COUNTPLOT
                # =================================================
                elif plot_type == "countplot":

                    if len(columns) == 1:

                        sns.countplot(
                            x=df[columns[0]]
                        )

                    elif len(columns) >= 2:

                        sns.countplot(
                            x=df[columns[0]],
                            hue=df[columns[1]]
                        )

                # =================================================
                # SCATTERPLOT
                # =================================================
                elif plot_type == "scatterplot":

                    if len(columns) >= 2:

                        sns.scatterplot(
                            x=df[columns[0]],
                            y=df[columns[1]]
                        )
                        
                plt.title(title)
                plt.tight_layout()
                plt.savefig(plot_path)
                plt.close()
                
                generated_plots.append({
                    "title": title,
                    "reason": reason,
                    "plot_type": plot_type,
                    "columns": columns,
                    "local_path": plot_path,
                    "frontend_path": relative_plot_path,
                    "filename": plot_filename
                })

            except Exception as viz_error:

                print(
                    f"Visualization Error: {viz_error}"
                )

        # =========================================================
        # APPEND GENERATED PLOTS TO JSON
        # =========================================================
        report_json["generated_plots"] = generated_plots

        json_path = os.path.join(
            output_dir,
            "eda_report.json"
        )

        with open(json_path, "w", encoding="utf-8") as f:

            json.dump(
                report_json,
                f,
                indent=4
            )

        print("\nEDA JSON GENERATED")
        print(f"Saved JSON: {json_path}")

        return {
            "status": "success",
            "json_report": json_path,
            "report": report_json
        }, data_path

    except Exception as exc:

        error_message = f"EDA Error: {exc}"

        print(error_message)

        return {
            "status": "error",
            "message": error_message
        }, data_path