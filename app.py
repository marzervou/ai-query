import os
import uuid
import streamlit as st
import pandas as pd
from databricks import sql
from databricks.sdk.core import Config
from pyspark.sql import SparkSession

# —— CONFIG & AUTH ——
cfg = Config()  # reads DATABRICKS_CLIENT_ID, CLIENT_SECRET, and host
HTTP_PATH = os.environ.get("DATABRICKS_SQL_ENDPOINT")
# Unity Catalog volume reference (e.g., "catalog.schema.volume_name")
VOLUME_NAME = os.environ.get("DATA_VOLUME_PATH")
if not VOLUME_NAME:
    st.error("Please configure the DATA_VOLUME_PATH env var to your UC volume, e.g. 'catalog.schema.volume_name'.")
    st.stop()

# Initialize Spark for writes to UC volume
spark = SparkSession.builder.getOrCreate()

# App-authenticated connection for write operations
def get_app_conn():
    return sql.connect(
        server_hostname=cfg.host,
        http_path=HTTP_PATH,
        credentials_provider=lambda: cfg.authenticate
    )

# User-authenticated connection for read operations
def get_user_conn():
    token = st.context.headers.get("X-Forwarded-Access-Token")
    if not token:
        st.error("User token missing: ensure 'sql' scope enabled under User Authorization")
        st.stop()
    return sql.connect(
        server_hostname=cfg.host,
        http_path=HTTP_PATH,
        access_token=token
    )

st.title("🚀 Batch AI Task Runner (Authorized)\nUsing UC Volume: {}".format(VOLUME_NAME))

# 1) Upload CSV
df = None
uploaded = st.file_uploader("🔽 Upload a CSV of texts", type="csv")
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview:", df.head())
else:
    st.stop()

# 2) Select column + AI task
text_col = st.selectbox("Select column to analyze", df.columns.tolist())
task = st.selectbox("Pick AI task", ["Sentiment Analysis", "Topic Classification", "Summarization"])
prompt_map = {
    "Sentiment Analysis":    'Determine the sentiment (positive/negative/neutral) of: "{}"',
    "Topic Classification":  'Classify the topic of: "{}"',
    "Summarization":         'Summarize the following text: "{}"',
}

# 3) Submit batch job
if st.button("▶️ Submit Batch Job"):
    job_id = uuid.uuid4().hex[:8]
    temp_table = f"user_upload_{job_id}"
    out_table  = f"user_upload_processed_{job_id}"

    # Convert Pandas DF to Spark and write to UC volume as Delta
    sdf = spark.createDataFrame(df)
    volume_path = f"@{VOLUME_NAME}/{temp_table}"
    sdf.write.mode("overwrite").format("delta").option("path", volume_path).saveAsTable(f"tmp.{temp_table}")

    # Run ai_query on the volume-based table
    app_conn = get_app_conn()
    sql_stmt = f"""
    CREATE OR REPLACE TABLE tmp.{out_table} USING DELTA LOCATION '{volume_path}' AS
    SELECT *, ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        format_string({repr(prompt_map[task])}, {text_col})
    ) AS ai_result
    FROM tmp.{temp_table};
    """
    app_conn.execute(sql_stmt)
    st.session_state["out_table"] = f"tmp.{out_table}"
    st.success(f"Batch job submitted! Output table: tmp.{out_table}")

# 4) Fetch results (user auth)
if "out_table" in st.session_state and st.button("📥 Fetch Results"):
    user_conn = get_user_conn()
    df_out = pd.read_sql(f"SELECT * FROM {st.session_state['out_table']}", user_conn)
    st.dataframe(df_out)

# 5) Optional evaluation (app auth)
if "out_table" in st.session_state and st.button("📝 Run Evaluation"):
    eval_table = st.session_state["out_table"].replace("processed", "evaluated")
    eval_sql = f"""
    CREATE OR REPLACE TABLE {eval_table} USING DELTA LOCATION '@{VOLUME_NAME}/{eval_table}' AS
    SELECT *, ai_query(
        'databricks-meta-llama-3-3-70b-instruct',
        format_string(
            'Evaluate this result: "%s". Is it accurate? Respond Yes/No with a brief explanation.',
            ai_result
        )
    ) AS evaluation
    FROM {st.session_state['out_table']};
    """
    get_app_conn().execute(eval_sql)
    st.success(f"Evaluation kicked off! Table: {eval_table}")
