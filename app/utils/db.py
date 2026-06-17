import os
import streamlit as st
import pandas as pd

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA  = "unstructured_data"
WAREHOUSE_ID = os.environ.get("WAREHOUSE_ID", "b04eb16e0536bd88")


def _hostname() -> str:
    """Return bare hostname (strips https:// if present — Databricks Apps injects the full URL)."""
    host = os.environ.get("DATABRICKS_HOST", "fevm-serverless-stable-r4umw1.cloud.databricks.com")
    return host.removeprefix("https://").removeprefix("http://").rstrip("/")


@st.cache_resource
def get_connection():
    from databricks import sql as dbsql
    token = os.environ.get("DATABRICKS_TOKEN")
    return dbsql.connect(
        server_hostname=_hostname(),
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        access_token=token,
    )


@st.cache_data(ttl=300)
def query(sql: str) -> pd.DataFrame:
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)


@st.cache_resource
def get_workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


@st.cache_data(ttl=3600, show_spinner=False)
def download_volume_file(path: str) -> bytes:
    """Download a file from a UC Volume path via Databricks Files API."""
    import urllib.request
    token = os.environ.get("DATABRICKS_TOKEN", "")
    # Use the Files API: GET /api/2.0/fs/files/{path}
    # path starts with /Volumes/... so strip the leading slash
    url = f"https://{_hostname()}/api/2.0/fs/files/{path.lstrip('/')}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req) as resp:
        return resp.read()


def full_table(table: str) -> str:
    return f"{CATALOG}.{SCHEMA}.{table}"


@st.cache_data(ttl=60)
def get_endpoint_statuses(endpoint_names: tuple) -> dict:
    """Returns {name: (ready: bool, state_str: str)} for each endpoint."""
    ws = get_workspace_client()
    statuses = {}
    for name in endpoint_names:
        try:
            ep = ws.serving_endpoints.get(name=name)
            ready = ep.state.ready.value == "READY" if ep.state and ep.state.ready else False
            state_str = ep.state.ready.value if ep.state and ep.state.ready else "UNKNOWN"
        except Exception as e:
            ready = False
            state_str = f"ERROR: {e}"
        statuses[name] = (ready, state_str)
    return statuses
