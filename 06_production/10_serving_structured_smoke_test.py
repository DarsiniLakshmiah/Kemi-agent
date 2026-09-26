# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 10 - Serving-compatible structured tool smoke test
# MAGIC This validates the SQL Statement Execution replacement for the notebook Spark Data Agent.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.102.0"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import sys

# ============================================================
# SQL Warehouse configuration
# ============================================================

SQL_WAREHOUSE_ID = "9b3ae64f1848fe3f"

assert SQL_WAREHOUSE_ID, "SQL Warehouse ID is required."

# Make it available to the serving-compatible runtime.
os.environ["DATABRICKS_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID

print("SQL Warehouse ID:", SQL_WAREHOUSE_ID)


# ============================================================
# Import production Data Agent runtime
# ============================================================

from pathlib import Path

# Project root = the folder containing serving_runtime/. Resolved relative to
# this notebook so it works from any Databricks Git folder or workspace path.
_notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook().getContext().notebookPath().get()
)
PROJECT_ROOT = str(Path("/Workspace" + _notebook_path).parent.parent)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from serving_runtime.data_agent_sql_runtime import get_indicator_series

print("data_agent_sql_runtime imported successfully.")


# ============================================================
# Test the production structured-data path
# ============================================================

result = get_indicator_series(
    entity_query="India",
    indicator_code="NY.GDP.MKTP.KD.ZG",
    start_year=2015,
    end_year=2025,
)

print("Query executed successfully.")
print("Resolved entity:", result["entity"])
print("Observation count:", len(result["observations"]))


# ============================================================
# Validate result
# ============================================================

assert result["entity"]["entity_id"] == "IND", (
    f"Expected IND but received: {result['entity']}"
)

assert result["start_year"] == 2015, (
    f"Unexpected start year: {result['start_year']}"
)

assert result["end_year"] == 2025, (
    f"Unexpected end year: {result['end_year']}"
)

assert len(result["observations"]) == 11, (
    f"Expected 11 observations, "
    f"received {len(result['observations'])}"
)

print("Serving-compatible structured tool PASSED.")
print(result)