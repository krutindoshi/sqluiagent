# SQL UI Agent

Upload CSV or Excel files and query them using SQL or English prompts.

Features:
- Streamlit UI
- DuckDB query engine
- Table explorer
- Join suggestions
- Chart visualization
- Download results
- AI support:
  - None
  - Ollama (free local)
  - OpenAI (user API key)

## Run locally

Install dependencies:

pip install -r requirements.txt

Run app:

streamlit run app.py

## Using Ollama (Free AI)

Install Ollama and run:

ollama run llama3.1

Then select "Ollama" inside the app.

## Using OpenAI

Select OpenAI in the app and paste your API key.
