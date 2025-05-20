import os
import streamlit as st
import pandas as pd
from databricks import sql
import uuid

# —— CONFIG ——
SERVER_HOSTNAME = os.environ["DATABRICKS_SERVER_HOSTNAME"]
HTTP_PATH       = os.environ["DATABRICKS_SQL_ENDPOINT"]
ACCESS_TOKEN    = os.environ["DATABRICKS_TOKEN"]

# Create a singleton connection
@st.experimental_singleton
def get_conn():
    return sql.connect(
        server_hostname=SERVER_HOSTNAME,
        http_path=HTTP_PATH,
        access_token=ACCESS_TOKEN
    )

conn = get_conn()

st.title("🚀 Batch AI Task Runner on Databricks")

# 1) Upload data
uploaded = st.file_uploader("🔽 Upload a CSV of your texts", type="csv")
if not uploaded:
    st.stop()

df = pd.read_csv(uploaded)
st.write("Preview of your data:", df.head())

# 2) Pick the text-column and AI task
text_col = st.selectbox("Select column to analyze", df.columns.tolist())
task     = st.selectbox("Pick AI task", [
    "Sentiment Analysis",
    "Topic Classification",
    "Summarization",
])
prompt_map = {
    "Sentiment Analysis":    'Determine the sentiment (positive/negative/neutral) of: "{txt}"',
    "Topic Classification":  'Classify the topic of: "{txt}"',
    "Summarization":         'Summarize the following text: "{txt}"',
}

if st.button("▶️ Submit Batch Job"):
    job_id = uuid.uuid4().hex[:8]
    temp_table = f"user_upload_{job_id}"
    out_table  = f"user_upload_processed_{job_id}"

    # 2a) Write the CSV into a temp Delta table
    df.to_parquet(f"/dbfs/tmp/{temp_table}.parquet", index=False)
    conn.execute(f"""
      CREATE OR REPLACE TABLE tmp.{temp_table}
      USING PARQUET
      LOCATION '/tmp/{temp_table}.parquet'
    """)

    # 2b) Launch the ai_query batch
    sql_stmt = f"""
    CREATE OR REPLACE TABLE tmp.{out_table} AS
    SELECT
      *,
      ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        format_string(
          '{prompt_map[task]}',
          {text_col}
        )
      ) AS ai_result
    FROM tmp.{temp_table};
    """
    conn.execute(sql_stmt)

    # store for later
    st.session_state["out_table"] = f"tmp.{out_table}"
    st.success(f"Batch job submitted! Output table: tmp.{out_table}")

# 3) Fetch & display results
if "out_table" in st.session_state:
    if st.button("📥 Fetch Results"):
        tbl = st.session_state["out_table"]
        df_out = pd.read_sql(f"SELECT * FROM {tbl}", conn)
        st.dataframe(df_out)

# 4) Optional evaluation step
if "out_table" in st.session_state:
    if st.button("📝 Run AI Evaluation"):
        tbl = st.session_state["out_table"]
        eval_tbl = tbl.replace("processed", "evaluated")
        eval_sql = f"""
        CREATE OR REPLACE TABLE {eval_tbl} AS
        SELECT
          *,
          ai_query(
            'databricks-meta-llama-3-3-70b-instruct',
            format_string(
              'Evaluate this result: "%s". Is it accurate? Respond Yes/No with a brief explanation.',
              ai_result
            )
          ) AS evaluation
        FROM {tbl};
        """
        conn.execute(eval_sql)
        st.success(f"Evaluation kicked off! Table: {eval_tbl}")
