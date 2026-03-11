import duckdb
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO


class DataTools:
    def __init__(self):
        self.conn = duckdb.connect()
        self.table_name = "data_table"
        self.df = None

    def load_csv(self, uploaded_file):
        self.df = pd.read_csv(uploaded_file)
        self.conn.register(self.table_name, self.df)
        return self.df

    def get_schema(self):
        schema_df = self.conn.execute(f"DESCRIBE {self.table_name}").fetchdf()
        return schema_df

    def preview_data(self, limit=5):
        return self.conn.execute(
            f"SELECT * FROM {self.table_name} LIMIT {limit}"
        ).fetchdf()

    def run_sql(self, sql):
        return self.conn.execute(sql).fetchdf()

    def get_column_list(self):
        if self.df is None:
            return []
        return list(self.df.columns)

    def create_chart(self, df, x_col=None, y_col=None):
        if df.empty or len(df.columns) < 2:
            return None

        if x_col is None:
            x_col = df.columns[0]
        if y_col is None:
            y_col = df.columns[1]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(df[x_col].astype(str), df[y_col])
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{y_col} by {x_col}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        buffer = BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        plt.close(fig)
        return buffer