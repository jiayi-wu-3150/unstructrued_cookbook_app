import os
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

CATALOG = "serverless_stable_r4umw1_catalog"
SCHEMA = "unstructured_data"

# App service principal authentication (matches official Databricks Apps template)
cfg = Config()


def _server_hostname() -> str:
    """Return bare hostname for SQL connector."""
    server_hostname = cfg.host
    if server_hostname.startswith("https://"):
        server_hostname = server_hostname.replace("https://", "")
    elif server_hostname.startswith("http://"):
        server_hostname = server_hostname.replace("http://", "")
    return server_hostname


@st.cache_resource
def get_connection():
    warehouse_id = os.environ.get("DATABRICKS_WAREHOUSE_ID", "b04eb16e0536bd88")
    return sql.connect(
        server_hostname=_server_hostname(),
        http_path=f"/sql/1.0/warehouses/{warehouse_id}",
        credentials_provider=lambda: cfg.authenticate,
        _use_arrow_native_complex_types=False,
    )


@st.cache_data(ttl=300)
def query(sql_str: str) -> pd.DataFrame:
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql_str)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)
    except Exception:
        # Connection may be stale (e.g. warehouse restarted) — clear cache and retry once
        get_connection.clear()
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute(sql_str)
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
        return pd.DataFrame(rows, columns=cols)


@st.cache_resource
def get_workspace_client():
    from databricks.sdk import WorkspaceClient
    return WorkspaceClient()


@st.cache_data(ttl=3600, show_spinner=False)
def download_volume_file(path: str) -> bytes:
    """Download a file from a UC Volume path via the Files API."""
    import urllib.request
    # Normalize path
    path = path.removeprefix("dbfs:").removeprefix("file:").lstrip("/")
    # Get auth headers from SDK Config
    headers = cfg.authenticate()
    url = f"https://{_server_hostname()}/api/2.0/fs/files/{path}"
    req = urllib.request.Request(url, headers=dict(headers))
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
