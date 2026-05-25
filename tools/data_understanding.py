import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from langchain_core.tools import tool

from jinja2 import Template


@tool
def data_understanding(task, tool_input, prompt, data_path, llm):
    """
    Perform intelligent exploratory data analysis and generate
    visual HTML reports using LLM-guided insights.
    """

    print("=========================================================================")

    df = pd.read_csv(data_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = f"output/dynamic_pipeline/{timestamp}"

    plots_dir = os.path.join(output_dir, "plots")

    os.makedirs(plots_dir, exist_ok=True)

    dataset_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_values": df.isnull().sum().to_dict()
    }

    eda_prompt = f"""
You are a senior data scientist.

Analyze the dataset and return ONLY valid JSON.

Do NOT return markdown.
Do NOT wrap JSON in code blocks.
Do NOT explain outside JSON.

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
    try:
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

            plot_type = viz.get("plot_type")

            columns = viz.get("columns", [])

            title = viz.get("title", "Visualization")

            reason = viz.get("reason", "")

            plot_filename = f"plot_{idx}.png"

            plot_path = os.path.join(
                plots_dir,
                plot_filename
            )

            try:

                plt.figure(figsize=(8, 5))

                if plot_type == "missing_values":

                    missing = df.isnull().sum()

                    missing = missing[missing > 0]

                    missing.sort_values(
                        ascending=False
                    ).plot(kind="bar")

                # ======================================
                # HEATMAP
                # ======================================
                elif plot_type == "heatmap":

                    numeric_df = df.select_dtypes(
                        include=["number"]
                    )

                    sns.heatmap(
                        numeric_df.corr(),
                        annot=True,
                        cmap="coolwarm"
                    )

                # ======================================
                # HISTOGRAM
                # ======================================
                elif plot_type == "histogram":

                    if columns:

                        sns.histplot(df[columns[0]])

                # ======================================
                # BOXPLOT
                # ======================================
                elif plot_type == "boxplot":

                    if columns:

                        sns.boxplot(
                            x=df[columns[0]]
                        )

                # ======================================
                # COUNTPLOT
                # ======================================
                elif plot_type == "countplot":

                    if columns:

                        sns.countplot(
                            x=df[columns[0]]
                        )

                # ======================================
                # SCATTERPLOT
                # ======================================
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
                    "path": f"plots/{plot_filename}"
                })

            except Exception as viz_error:

                print(
                    f"Visualization Error: {viz_error}"
                )
                
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

        # =========================================================
        # HTML TEMPLATE
        # =========================================================
        html_template = """
<!DOCTYPE html>
<html>

<head>

<meta charset="UTF-8">

<title>{{ title }}</title>

<style>

body{
    font-family: Arial;
    padding:40px;
    background:#f4f4f4;
    color:#222;
}

.container{
    max-width:1200px;
    margin:auto;
}

.card{
    background:white;
    padding:25px;
    margin-bottom:25px;
    border-radius:12px;
    box-shadow:0 2px 8px rgba(0,0,0,0.1);
}

h1{
    color:#111;
}

h2{
    color:#333;
}

img{
    width:100%;
    border-radius:10px;
    margin-top:15px;
}

.recommendation{
    padding:10px;
    background:#eef6ff;
    border-left:5px solid #007bff;
    margin-bottom:10px;
}

.summary{
    font-size:18px;
    line-height:1.7;
}
.section-text{
    line-height:1.8;
    margin-bottom:15px;
}

.bullet-item{
    padding:10px;
    margin-bottom:10px;
    background:#fafafa;
    border-radius:8px;
}

.warning-box{
    background:#fff4e5;
    border-left:5px solid #ff9800;
    padding:15px;
    margin-top:10px;
    margin-bottom:10px;
    border-radius:8px;
}

.metric-card{
    background:#f7faff;
    padding:15px;
    border-radius:10px;
    margin-top:10px;
    margin-bottom:10px;
    border:1px solid #dbeafe;
}

.metric-label{
    font-size:14px;
    color:#666;
}

.metric-value{
    font-size:24px;
    font-weight:bold;
    margin-top:5px;
}
</style>

</head>

<body>

<div class="container">

<div class="card">

<h1>{{ title }}</h1>

<p class="summary">
{{ summary }}
</p>

</div>

{% for section in sections %}

<div class="card">

<h2>{{ section.title }}</h2>

{% for item in section.content %}

    {% if item.type == "text" %}

        <p class="section-text">
            {{ item.value }}
        </p>

    {% elif item.type == "bullet" %}

        <div class="bullet-item">
            • {{ item.value }}
        </div>

    {% elif item.type == "warning" %}

        <div class="warning-box">
            ⚠ {{ item.value }}
        </div>

    {% elif item.type == "metric" %}

        <div class="metric-card">

            <div class="metric-label">
                {{ item.label }}
            </div>

            <div class="metric-value">
                {{ item.value }}
            </div>

        </div>

    {% endif %}

{% endfor %}

</div>

{% endfor %}

{% for plot in plots %}

<div class="card">

<h2>{{ plot.title }}</h2>

<p>{{ plot.reason }}</p>

<img src="{{ plot.path }}">

</div>

{% endfor %}

<div class="card">

<h2>Recommendations</h2>

{% for rec in recommendations %}

<div class="recommendation">

{{ rec }}

</div>

{% endfor %}

</div>

</div>

</body>
</html>
"""
        template = Template(html_template)

        rendered_html = template.render(
            title=report_json.get("title", "EDA Report"),
            summary=report_json.get("summary", ""),
            sections=report_json.get("sections", []),
            plots=generated_plots,
            recommendations=report_json.get(
                "recommendations",
                []
            )
        )

        html_path = os.path.join(
            output_dir,
            "eda_report.html"
        )

        with open(html_path, "w", encoding="utf-8") as f:

            f.write(rendered_html)

        print("\nEDA REPORT GENERATED")

        print(f"Saved HTML: {html_path}")

        print(f"Saved JSON: {json_path}")

        return {
            "status": "success",
            "html_report": html_path,
            "json_report": json_path,
            "summary": report_json.get("summary", "")
        }, data_path

    except Exception as exc:

        error_message = f"EDA Error: {exc}"

        print(error_message)

        return {
            "status": "error",
            "message": error_message
        }, data_path