import os
import re
import json
import requests
import duckdb
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

st.set_page_config(page_title="Data Workspace", layout="wide")

st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}
div[data-testid="stMetric"] {
    background: #f7f7f7;
    padding: 10px 14px;
    border-radius: 10px;
}
div.stButton > button {
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

st.title("Data Workspace")
st.caption("Upload CSV/Excel files, explore tables, write SQL, or use AI to generate queries.")

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "conn" not in st.session_state:
    st.session_state.conn = duckdb.connect()

if "tables" not in st.session_state:
    st.session_state.tables = {}

if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "generated_sql" not in st.session_state:
    st.session_state.generated_sql = ""

if "generated_explanation" not in st.session_state:
    st.session_state.generated_explanation = ""

if "explorer_table" not in st.session_state:
    st.session_state.explorer_table = None

if "page_mode" not in st.session_state:
    st.session_state.page_mode = "Workspace"

if "active_table" not in st.session_state:
    st.session_state.active_table = None

if "provider" not in st.session_state:
    st.session_state.provider = "None"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def sanitize_table_name(name: str) -> str:
    base = os.path.splitext(name)[0].lower()
    base = re.sub(r"[^a-zA-Z0-9_]", "_", base)
    if re.match(r"^\d", base):
        base = f"t_{base}"
    return base


def load_file_to_duckdb(uploaded_file, conn):
    filename = uploaded_file.name
    table_name = sanitize_table_name(filename)

    if filename.lower().endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif filename.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Unsupported file type: {filename}")

    conn.register(f"{table_name}_tmp", df)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM {table_name}_tmp")
    conn.unregister(f"{table_name}_tmp")
    return table_name, df


def get_schema_info(conn, table_name):
    return conn.execute(f"DESCRIBE {table_name}").fetchdf()


def list_columns(conn, table_name):
    schema_df = get_schema_info(conn, table_name)
    return schema_df["column_name"].tolist()


def get_all_schema_text(conn, tables):
    parts = []
    for table_name in tables:
        schema_df = get_schema_info(conn, table_name)
        parts.append(f"Table: {table_name}\n{schema_df.to_string(index=False)}\n")
    return "\n".join(parts)


def is_safe_sql(sql: str) -> bool:
    banned = [
        "DROP ", "DELETE ", "TRUNCATE ", "ALTER ", "UPDATE ",
        "INSERT ", "MERGE ", "REPLACE ", "CREATE ", "ATTACH ",
        "DETACH ", "COPY ", "CALL "
    ]
    sql_upper = sql.upper().strip()
    if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
        return False
    return not any(word in sql_upper for word in banned)


def add_history(sql_text):
    st.session_state.history.insert(0, sql_text)
    st.session_state.history = st.session_state.history[:20]


def get_matching_column(columns, keywords):
    for kw in keywords:
        for col in columns:
            if kw in col.lower():
                return col
    return None


def fallback_nl_to_sql(question, tables):
    q = question.lower().strip()
    active_table = st.session_state.active_table or (list(tables.keys())[0] if tables else None)

    if not active_table:
        return "SELECT 1"

    columns = tables[active_table]["columns"]

    if "top 10" in q or "sample" in q or "preview" in q:
        return f"SELECT * FROM {active_table} LIMIT 10"

    if "show data" in q or "all rows" in q:
        return f"SELECT * FROM {active_table} LIMIT 100"

    if "count rows" in q or q == "count" or "number of rows" in q or "row count" in q:
        return f"SELECT COUNT(*) AS row_count FROM {active_table}"

    if "distinct" in q:
        col = get_matching_column(columns, ["country", "category", "customer", "state", "city"])
        if not col and columns:
            col = columns[0]
        return f"SELECT DISTINCT {col} FROM {active_table} LIMIT 100"

    amount_col = get_matching_column(columns, ["amount", "sales", "revenue", "price", "cost", "spend"])
    country_col = get_matching_column(columns, ["country"])
    category_col = get_matching_column(columns, ["category", "segment", "type"])
    customer_col = get_matching_column(columns, ["customer", "client", "name", "user"])
    date_col = get_matching_column(columns, ["date", "order_date", "created", "timestamp"])

    if ("total" in q or "sum" in q) and "country" in q and amount_col and country_col:
        return f"""
SELECT {country_col}, SUM({amount_col}) AS total_value
FROM {active_table}
GROUP BY {country_col}
ORDER BY total_value DESC
""".strip()

    if ("total" in q or "sum" in q) and "category" in q and amount_col and category_col:
        return f"""
SELECT {category_col}, SUM({amount_col}) AS total_value
FROM {active_table}
GROUP BY {category_col}
ORDER BY total_value DESC
""".strip()

    if ("total" in q or "sum" in q) and "customer" in q and amount_col and customer_col:
        return f"""
SELECT {customer_col}, SUM({amount_col}) AS total_value
FROM {active_table}
GROUP BY {customer_col}
ORDER BY total_value DESC
""".strip()

    if ("average" in q or "avg" in q) and "country" in q and amount_col and country_col:
        return f"""
SELECT {country_col}, AVG({amount_col}) AS avg_value
FROM {active_table}
GROUP BY {country_col}
ORDER BY avg_value DESC
""".strip()

    if ("average" in q or "avg" in q) and "category" in q and amount_col and category_col:
        return f"""
SELECT {category_col}, AVG({amount_col}) AS avg_value
FROM {active_table}
GROUP BY {category_col}
ORDER BY avg_value DESC
""".strip()

    if ("highest" in q or "top customer" in q or "highest spend" in q or "most spend" in q) and amount_col and customer_col:
        return f"""
SELECT {customer_col}, SUM({amount_col}) AS total_value
FROM {active_table}
GROUP BY {customer_col}
ORDER BY total_value DESC
LIMIT 10
""".strip()

    if ("sales by month" in q or "revenue by month" in q or "amount by month" in q) and amount_col and date_col:
        return f"""
SELECT DATE_TRUNC('month', {date_col}) AS month, SUM({amount_col}) AS total_value
FROM {active_table}
GROUP BY month
ORDER BY month
""".strip()

    if ("top 5" in q or "highest 5" in q) and amount_col:
        return f"""
SELECT *
FROM {active_table}
ORDER BY {amount_col} DESC
LIMIT 5
""".strip()

    return f"SELECT * FROM {active_table} LIMIT 10"


def generate_sql_openai(question, schema_text, api_key):
    client = OpenAI(api_key=api_key)

    prompt = f"""
You are a data analyst that writes DuckDB SQL.

Rules:
- Return ONLY valid JSON
- Use only the tables and columns in the schema
- Only write read-only SQL
- SQL must begin with SELECT or WITH
- No markdown

Return exactly:
{{
  "sql": "...",
  "explanation": "..."
}}

Schema:
{schema_text}

User question:
{question}
"""

    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )
    raw = response.output_text.strip()
    data = json.loads(raw)
    return data["sql"], data.get("explanation", "")


def generate_sql_ollama(question, schema_text, model_name):
    prompt = f"""
You are a data analyst that writes DuckDB SQL.

Rules:
- Return only JSON
- Use only the tables and columns in the schema
- Only read-only SQL
- SQL must begin with SELECT or WITH
- No markdown

Return exactly:
{{
  "sql": "...",
  "explanation": "..."
}}

Schema:
{schema_text}

User question:
{question}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    response.raise_for_status()
    raw = response.json()["response"].strip()
    data = json.loads(raw)
    return data["sql"], data.get("explanation", "")


def summarize_result_openai(question, result_df, api_key):
    client = OpenAI(api_key=api_key)
    prompt = f"""
You are a business data analyst.

User question:
{question}

Result:
{result_df.head(20).to_string(index=False)}

Give a concise 3 bullet summary.
"""
    response = client.responses.create(
        model="gpt-5.4",
        input=prompt
    )
    return response.output_text.strip()


def summarize_result_ollama(question, result_df, model_name):
    prompt = f"""
You are a business data analyst.

User question:
{question}

Result:
{result_df.head(20).to_string(index=False)}

Give a concise 3 bullet summary.
"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_name,
            "prompt": prompt,
            "stream": False
        },
        timeout=60
    )
    response.raise_for_status()
    return response.json()["response"].strip()


def draw_chart(df, x_col, y_col, chart_type):
    fig, ax = plt.subplots(figsize=(8, 4))

    if chart_type == "Bar":
        ax.bar(df[x_col].astype(str), df[y_col])
    elif chart_type == "Line":
        ax.plot(df[x_col].astype(str), df[y_col])
    elif chart_type == "Scatter":
        ax.scatter(df[x_col], df[y_col])

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} by {x_col}")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def suggest_joins(tables_meta):
    suggestions = []
    table_names = list(tables_meta.keys())

    for i in range(len(table_names)):
        for j in range(i + 1, len(table_names)):
            t1 = table_names[i]
            t2 = table_names[j]
            cols1 = set(tables_meta[t1]["columns"])
            cols2 = set(tables_meta[t2]["columns"])
            shared = cols1.intersection(cols2)
            likely = [c for c in shared if c.endswith("_id") or c in ["id", "date"]]

            for col in likely[:3]:
                suggestions.append((t1, t2, col))

    return suggestions


def set_sql_and_go(sql_text):
    st.session_state.generated_sql = sql_text
    st.session_state.page_mode = "Workspace"


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.header("Settings")

    provider_options = ["None", "OpenAI", "Ollama (Local Only)"]
    current_provider = st.session_state.provider if st.session_state.provider in provider_options else "None"

    st.session_state.provider = st.selectbox(
        "AI Provider",
        provider_options,
        index=provider_options.index(current_provider)
    )

    openai_api_key = ""
    ollama_model = "llama3.1"

    if st.session_state.provider == "OpenAI":
        openai_api_key = st.text_input(
            "Enter OpenAI API Key",
            type="password",
            placeholder="sk-..."
        )
        st.caption("Your key is used only for this session.")

    elif st.session_state.provider == "Ollama (Local Only)":
        ollama_model = st.text_input(
            "Ollama model",
            value="llama3.1"
        )
        st.caption("Run Ollama locally first, for example: ollama run llama3.1")
        st.warning("This option will not work on hosted Streamlit Cloud.")

    else:
        st.info("None mode supports basic prompts like top 10 rows, count rows, total sales by country, and average amount by category.")

    st.markdown("---")
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader(
        "Upload CSV / Excel files",
        type=["csv", "xlsx", "xls"],
        accept_multiple_files=True
    )

    if uploaded_files:
        for f in uploaded_files:
            try:
                table_name, df = load_file_to_duckdb(f, st.session_state.conn)
                st.session_state.tables[table_name] = {
                    "file_name": f.name,
                    "rows": len(df),
                    "columns": list(df.columns)
                }

                if st.session_state.explorer_table is None:
                    st.session_state.explorer_table = table_name
                if st.session_state.active_table is None:
                    st.session_state.active_table = table_name
            except Exception as e:
                st.error(f"Failed to load {f.name}: {e}")

    st.markdown("---")
    st.subheader("Catalog")

    if st.session_state.tables:
        for table_name, meta in st.session_state.tables.items():
            with st.container():
                st.markdown(f"**{table_name}**")
                st.caption(f"{meta['rows']} rows • {len(meta['columns'])} columns")

                c1, c2 = st.columns(2)
                with c1:
                    if st.button("Explore", key=f"explore_{table_name}", use_container_width=True):
                        st.session_state.explorer_table = table_name
                        st.session_state.page_mode = "Explorer"
                        st.rerun()
                with c2:
                    if st.button("Use", key=f"use_{table_name}", use_container_width=True):
                        st.session_state.active_table = table_name
                        set_sql_and_go(f"SELECT * FROM {table_name} LIMIT 10")
                        st.rerun()
    else:
        st.caption("No tables loaded.")

    st.markdown("---")
    st.subheader("History")
    if st.session_state.history:
        for i, q in enumerate(st.session_state.history[:8], start=1):
            st.caption(f"{i}. {q[:60]}")
    else:
        st.caption("No query history yet.")


# --------------------------------------------------
# Top nav
# --------------------------------------------------
nav1, nav2 = st.columns([1, 4])

with nav1:
    current_index = ["Workspace", "Explorer", "Joins"].index(st.session_state.page_mode) \
        if st.session_state.page_mode in ["Workspace", "Explorer", "Joins"] else 0

    st.session_state.page_mode = st.radio(
        "View",
        ["Workspace", "Explorer", "Joins"],
        index=current_index
    )

with nav2:
    if st.session_state.page_mode == "Workspace":
        st.caption("Query your data with SQL or English prompts.")
    elif st.session_state.page_mode == "Explorer":
        st.caption("Inspect schema and preview table data.")
    else:
        st.caption("Review likely join relationships across uploaded tables.")


# --------------------------------------------------
# Workspace
# --------------------------------------------------
if st.session_state.page_mode == "Workspace":
    table_names = list(st.session_state.tables.keys())

    header1, header2, header3 = st.columns([4, 1, 1])

    with header1:
        question = st.text_input(
            "Ask in English",
            value="Show top 10 rows",
            placeholder="Example: Show total sales by country ordered from highest to lowest"
        )

    with header2:
        active_table = st.selectbox(
            "Active table",
            table_names if table_names else ["No table loaded"],
            index=table_names.index(st.session_state.active_table)
            if st.session_state.active_table in table_names else 0
        )
        if active_table != "No table loaded":
            st.session_state.active_table = active_table

    with header3:
        st.write("")
        st.write("")
        generate_clicked = st.button("Generate SQL", use_container_width=True)

    action1, action2, action3, action4 = st.columns([1, 1, 1, 1])

    with action1:
        run_clicked = st.button("Run Query", use_container_width=True)
    with action2:
        clear_history = st.button("Clear History", use_container_width=True)
    with action3:
        helper_preview = st.button("Top 10", use_container_width=True)
    with action4:
        helper_count = st.button("Count Rows", use_container_width=True)

    if clear_history:
        st.session_state.history = []
        st.success("History cleared.")

    if helper_preview and st.session_state.active_table:
        st.session_state.generated_sql = f"SELECT * FROM {st.session_state.active_table} LIMIT 10"

    if helper_count and st.session_state.active_table:
        st.session_state.generated_sql = f"SELECT COUNT(*) AS row_count FROM {st.session_state.active_table}"

    helper_cols = st.columns(4)
    with helper_cols[0]:
        helper_distinct = st.button("Distinct First Col", use_container_width=True)
    with helper_cols[1]:
        helper_nulls = st.button("Null Check", use_container_width=True)
    with helper_cols[2]:
        helper_schema = st.button("Preview Schema", use_container_width=True)
    with helper_cols[3]:
        helper_reset = st.button("Reset SQL", use_container_width=True)

    if st.session_state.active_table and helper_distinct:
        cols = st.session_state.tables[st.session_state.active_table]["columns"]
        if cols:
            st.session_state.generated_sql = f"SELECT DISTINCT {cols[0]} FROM {st.session_state.active_table} LIMIT 100"

    if st.session_state.active_table and helper_nulls:
        cols = st.session_state.tables[st.session_state.active_table]["columns"]
        if cols:
            checks = ",\n".join(
                [f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}_nulls" for c in cols[:8]]
            )
            st.session_state.generated_sql = f"SELECT\n{checks}\nFROM {st.session_state.active_table}"

    if st.session_state.active_table and helper_schema:
        st.session_state.page_mode = "Explorer"
        st.session_state.explorer_table = st.session_state.active_table
        st.rerun()

    if helper_reset and st.session_state.active_table:
        st.session_state.generated_sql = f"SELECT * FROM {st.session_state.active_table} LIMIT 10"

    if generate_clicked:
        if not st.session_state.tables:
            st.error("Upload at least one file first.")
        else:
            schema_text = get_all_schema_text(st.session_state.conn, st.session_state.tables)

            try:
                if st.session_state.provider == "OpenAI":
                    if not OPENAI_AVAILABLE:
                        st.error("OpenAI package not installed. Run: pip install openai")
                    elif not openai_api_key:
                        st.error("Enter an OpenAI API key in the left sidebar.")
                    else:
                        sql, explanation = generate_sql_openai(question, schema_text, openai_api_key)
                        st.session_state.generated_sql = sql
                        st.session_state.generated_explanation = explanation

                elif st.session_state.provider == "Ollama (Local Only)":
                    try:
                        sql, explanation = generate_sql_ollama(question, schema_text, ollama_model)
                        st.session_state.generated_sql = sql
                        st.session_state.generated_explanation = explanation
                    except Exception as e:
                        st.error(f"Ollama request failed: {e}")

                else:
                    st.session_state.generated_sql = fallback_nl_to_sql(question, st.session_state.tables)
                    st.session_state.generated_explanation = "Generated using built-in fallback logic."

            except Exception as e:
                st.error(f"Failed to generate SQL: {e}")

    if not st.session_state.generated_sql and st.session_state.active_table:
        st.session_state.generated_sql = f"SELECT * FROM {st.session_state.active_table} LIMIT 10"

    left, right = st.columns([1.05, 1.35])

    with left:
        st.subheader("SQL Worksheet")
        sql_text = st.text_area(
            "Edit SQL",
            value=st.session_state.generated_sql,
            height=260
        )

        if st.session_state.generated_explanation:
            st.caption(st.session_state.generated_explanation)

        with st.expander("Query History", expanded=False):
            if st.session_state.history:
                for q in st.session_state.history:
                    st.code(q, language="sql")
            else:
                st.caption("No query history yet.")

    with right:
        if run_clicked:
            if not st.session_state.tables:
                st.error("Upload files first.")
            elif not is_safe_sql(sql_text):
                st.error("Only read-only SELECT/WITH queries are allowed.")
            else:
                try:
                    result_df = st.session_state.conn.execute(sql_text).fetchdf()
                    st.session_state.last_result = result_df
                    st.session_state.generated_sql = sql_text
                    add_history(sql_text)
                    st.success("Query executed successfully.")
                except Exception as e:
                    st.error(f"Query failed: {e}")

        st.subheader("Results")
        tabs = st.tabs(["Data", "Chart", "Summary", "Download"])

        with tabs[0]:
            if st.session_state.last_result is not None:
                rows_count = len(st.session_state.last_result)
                cols_count = len(st.session_state.last_result.columns)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Rows Returned", rows_count)
                with m2:
                    st.metric("Columns Returned", cols_count)
                with m3:
                    st.metric("Active Table", st.session_state.active_table or "-")

                st.dataframe(st.session_state.last_result, use_container_width=True, height=380)
            else:
                st.info("Run a query to see results here.")

        with tabs[1]:
            result_df = st.session_state.last_result
            if result_df is not None and not result_df.empty and len(result_df.columns) >= 2:
                numeric_cols = result_df.select_dtypes(include="number").columns.tolist()
                all_cols = result_df.columns.tolist()

                if numeric_cols:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        x_col = st.selectbox("X-axis", all_cols, index=0, key="chart_x")
                    with c2:
                        y_col = st.selectbox(
                            "Y-axis",
                            all_cols,
                            index=all_cols.index(numeric_cols[0]),
                            key="chart_y"
                        )
                    with c3:
                        chart_type = st.selectbox("Chart", ["Bar", "Line", "Scatter"], key="chart_type")

                    if st.button("Draw Chart", use_container_width=True):
                        draw_chart(result_df, x_col, y_col, chart_type)
                else:
                    st.info("Need at least one numeric column for charting.")
            else:
                st.info("Run a query that returns at least two columns.")

        with tabs[2]:
            result_df = st.session_state.last_result
            if result_df is not None and not result_df.empty:
                if st.session_state.provider == "OpenAI" and openai_api_key and OPENAI_AVAILABLE:
                    if st.button("Generate AI Summary", use_container_width=True):
                        try:
                            summary = summarize_result_openai(question, result_df, openai_api_key)
                            st.write(summary)
                        except Exception as e:
                            st.error(f"Summary failed: {e}")
                elif st.session_state.provider == "Ollama (Local Only)":
                    if st.button("Generate AI Summary", use_container_width=True):
                        try:
                            summary = summarize_result_ollama(question, result_df, ollama_model)
                            st.write(summary)
                        except Exception as e:
                            st.error(f"Summary failed: {e}")
                else:
                    st.write(f"- Returned {len(result_df)} rows")
                    st.write(f"- Columns: {', '.join(result_df.columns)}")
                    st.write("- Enable OpenAI or local Ollama for an AI-generated summary.")
            else:
                st.info("Run a query first.")

        with tabs[3]:
            result_df = st.session_state.last_result
            if result_df is not None:
                csv_data = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download CSV",
                    data=csv_data,
                    file_name="query_result.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                st.info("Run a query first to download results.")


# --------------------------------------------------
# Explorer
# --------------------------------------------------
elif st.session_state.page_mode == "Explorer":
    st.subheader("Table Explorer")

    if st.session_state.tables and st.session_state.explorer_table:
        current_table = st.session_state.explorer_table
        current_meta = st.session_state.tables[current_table]

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Table", current_table)
        with m2:
            st.metric("Rows", current_meta["rows"])
        with m3:
            st.metric("Columns", len(current_meta["columns"]))

        explorer_tabs = st.tabs(["Schema", "Data Preview", "Columns"])

        with explorer_tabs[0]:
            schema_df = get_schema_info(st.session_state.conn, current_table)
            st.dataframe(schema_df, use_container_width=True, height=380)

        with explorer_tabs[1]:
            preview_df = st.session_state.conn.execute(
                f"SELECT * FROM {current_table} LIMIT 20"
            ).fetchdf()
            st.dataframe(preview_df, use_container_width=True, height=480)

        with explorer_tabs[2]:
            cols = current_meta["columns"]
            st.code(", ".join(cols))
            if st.button("Use This Table in Workspace", use_container_width=True):
                st.session_state.active_table = current_table
                st.session_state.generated_sql = f"SELECT * FROM {current_table} LIMIT 10"
                st.session_state.page_mode = "Workspace"
                st.rerun()
    else:
        st.info("Load a file and click Explore in the left sidebar.")


# --------------------------------------------------
# Joins
# --------------------------------------------------
elif st.session_state.page_mode == "Joins":
    st.subheader("Join Suggestions")

    joins = suggest_joins(st.session_state.tables)

    if joins:
        for i, (t1, t2, col) in enumerate(joins, start=1):
            st.markdown(f"**Suggestion {i}:** `{t1}` ↔ `{t2}` on `{col}`")

            suggested_sql = f"""
SELECT *
FROM {t1} a
JOIN {t2} b
  ON a.{col} = b.{col}
LIMIT 100
""".strip()

            st.code(suggested_sql, language="sql")

            if st.button(f"Use Join {i}", key=f"use_join_{i}", use_container_width=True):
                st.session_state.generated_sql = suggested_sql
                st.session_state.page_mode = "Workspace"
                st.rerun()
    else:
        st.info("No likely joins found yet. Upload more files with shared keys like customer_id, order_id, or id.")