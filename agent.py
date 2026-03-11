import json
from openai import OpenAI


class DataAgent:
    def __init__(self, api_key=None):
        self.client = OpenAI(api_key=api_key)

    def build_prompt(self, question, schema_text, preview_text):
        return f"""
You are a data analysis agent.

Your job:
1. Understand the user's question.
2. Write a correct DuckDB SQL query using the table name data_table.
3. Suggest whether the result should be visualized.
4. Explain the intent briefly.

Rules:
- Use only the table name: data_table
- Return ONLY valid JSON
- Do not wrap JSON in markdown
- SQL must be syntactically valid DuckDB SQL
- If aggregation is needed, use clear aliases
- If the question is ambiguous, make the most reasonable assumption

Return JSON in this exact format:
{{
  "intent": "...",
  "sql": "...",
  "show_chart": true,
  "chart_hint": {{
    "x": "column_name",
    "y": "column_name"
  }}
}}

Schema:
{schema_text}

Sample rows:
{preview_text}

User question:
{question}
"""

    def generate_plan(self, question, schema_df, preview_df):
        schema_text = schema_df.to_string(index=False)
        preview_text = preview_df.to_string(index=False)

        prompt = self.build_prompt(question, schema_text, preview_text)

        response = self.client.responses.create(
            model="gpt-5.4",
            input=prompt
        )

        raw_text = response.output_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            return {
                "intent": "Failed to parse structured response.",
                "sql": "SELECT * FROM data_table LIMIT 10",
                "show_chart": False,
                "chart_hint": {"x": "", "y": ""}
            }

    def summarize_result(self, question, result_df):
        if result_df.empty:
            return "The query returned no rows."

        prompt = f"""
You are a data analyst.

User question:
{question}

Query result:
{result_df.head(20).to_string(index=False)}

Provide a concise business-friendly explanation in 3-5 lines.
"""

        response = self.client.responses.create(
            model="gpt-5.4",
            input=prompt
        )

        return response.output_text.strip()