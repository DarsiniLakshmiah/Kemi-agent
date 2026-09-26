# Databricks notebook source
# MAGIC %md
# MAGIC # 11_mlflow_package_register_v3
# MAGIC
# MAGIC Production MLflow packaging and Unity Catalog registration for the World Bank GEP Intelligence Agent.
# MAGIC
# MAGIC - `serving_runtime` is located automatically relative to this notebook, so it works from a Databricks Git folder.
# MAGIC - `serving_runtime` is placed directly under MLflow `code/`.
# MAGIC - Every Databricks resource the agent calls is declared via `resources=`, so Model Serving provisions credentials for the endpoint (automatic authentication passthrough). Without this the serving container has no credentials and the endpoint update fails.
# MAGIC - The logged artifact is loaded in isolation before it is registered.

# COMMAND ----------

# MAGIC %pip install -q "mlflow>=3.12.0" "databricks-sdk>=0.102.0" databricks-openai databricks-ai-search pandas pydantic

# COMMAND ----------

# Restart Python so the packages installed above are the ones imported below.
dbutils.library.restartPython()

# COMMAND ----------

# CELL 2 — Imports and configuration
from pathlib import Path
import os
import shutil
import sys
import importlib
import re

import mlflow
import pandas as pd
from mlflow.models import infer_signature
from mlflow.models.resources import (
    DatabricksServingEndpoint,
    DatabricksSQLWarehouse,
    DatabricksTable,
    DatabricksVectorSearchIndex,
)

MODEL_NAME = "worldbank_ai.ai.gep_intelligence_agent"

# Databricks resources the serving runtime calls. These must match the
# constants in serving_runtime/*.py.
SQL_WAREHOUSE_ID = "9b3ae64f1848fe3f"
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
EMBEDDING_ENDPOINT = "databricks-qwen3-embedding-0-6b"
AI_SEARCH_INDEX = "worldbank_ai.rag.gep_structure_v1_qwen3_index"
MACRO_TABLE = "worldbank_ai.gold.macroeconomic_indicators"

# The structured Data Agent reads this at call time; set it for the
# pre-registration prediction test below.
os.environ["DATABRICKS_SQL_WAREHOUSE_ID"] = SQL_WAREHOUSE_ID


def find_repo_root() -> Path:
    """Locate the project root (the folder that contains serving_runtime/).

    Works for a Databricks Git folder, a Repos checkout, or a plain
    workspace upload, without hardcoding a user-specific path.
    """
    candidates = [Path.cwd(), Path.cwd().parent]
    try:
        notebook_path = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook().getContext().notebookPath().get()
        )
        notebook_dir = Path("/Workspace" + notebook_path).parent
        candidates += [notebook_dir, notebook_dir.parent]
    except Exception:
        pass

    for candidate in candidates:
        if (candidate / "serving_runtime" / "worldbank_model.py").exists():
            return candidate

    raise FileNotFoundError(
        "Could not find serving_runtime/worldbank_model.py. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )


REPO_ROOT = find_repo_root()
SOURCE_DIR = REPO_ROOT / "serving_runtime"

PACKAGE_ROOT = Path("/tmp/worldbank_model_package_v3")
PACKAGE_DIR = PACKAGE_ROOT / "serving_runtime"

RUNTIME_FILES = [
    "supervisor_runtime.py",
    "data_agent_sql_runtime.py",
    "research_agent_runtime.py",
    "synthesis_agent_runtime.py",
    "guardrails_runtime.py",
    "orchestrator.py",
    "worldbank_model.py",
]

print("Model:", MODEL_NAME)
print("Source:", SOURCE_DIR)
print("Package directory:", PACKAGE_DIR)

# COMMAND ----------

# CELL 3 — Build a real Python package
missing = [name for name in RUNTIME_FILES if not (SOURCE_DIR / name).exists()]
if missing:
    raise FileNotFoundError(f"Missing runtime files: {missing}")

if PACKAGE_ROOT.exists():
    shutil.rmtree(PACKAGE_ROOT)
PACKAGE_DIR.mkdir(parents=True, exist_ok=True)

for filename in RUNTIME_FILES:
    src = SOURCE_DIR / filename
    dst = PACKAGE_DIR / filename
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"COPIED: {filename}")

(PACKAGE_DIR / "__init__.py").write_text(
    "# World Bank GEP production serving runtime package.\n",
    encoding="utf-8",
)
print("CREATED: __init__.py")

# COMMAND ----------

# CELL 4 — Normalize local imports in packaged COPY only
orch_path = PACKAGE_DIR / "orchestrator.py"
orch = orch_path.read_text(encoding="utf-8")
replacements = {
    "from supervisor_runtime import": "from serving_runtime.supervisor_runtime import",
    "from data_agent_sql_runtime import": "from serving_runtime.data_agent_sql_runtime import",
    "from research_agent_runtime import": "from serving_runtime.research_agent_runtime import",
    "from synthesis_agent_runtime import": "from serving_runtime.synthesis_agent_runtime import",
    "from guardrails_runtime import": "from serving_runtime.guardrails_runtime import",
}
for old, new in replacements.items():
    orch = orch.replace(old, new)
orch_path.write_text(orch, encoding="utf-8")

model_path = PACKAGE_DIR / "worldbank_model.py"
model_text = model_path.read_text(encoding="utf-8")
model_text = model_text.replace(
    "from orchestrator import execute_agent",
    "from serving_runtime.orchestrator import execute_agent",
)
model_path.write_text(model_text, encoding="utf-8")
print("Normalized package imports")

# COMMAND ----------

# CELL 5 — Static validation of local imports
local_modules = {Path(name).stem for name in RUNTIME_FILES}
problems = []
for path in sorted(PACKAGE_DIR.glob("*.py")):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        for module in local_modules:
            if re.match(rf"^from\s+{re.escape(module)}\s+import", stripped):
                problems.append((path.name, line_no, stripped))
            if re.match(rf"^import\s+{re.escape(module)}(?:\s|$)", stripped):
                problems.append((path.name, line_no, stripped))
if problems:
    for problem in problems:
        print("UNRESOLVED:", problem)
    raise RuntimeError("Unresolved top-level local imports remain")
print("PASS: no unresolved top-level local imports")

# COMMAND ----------

# CELL 6 — Import-test staged package
package_root_str = str(PACKAGE_ROOT)
sys.path.insert(0, package_root_str) if package_root_str not in sys.path else None
for name in list(sys.modules):
    if name == "serving_runtime" or name.startswith("serving_runtime."):
        del sys.modules[name]
importlib.invalidate_caches()

import serving_runtime
print("Package loaded from:", serving_runtime.__file__)
from serving_runtime.orchestrator import execute_agent
print("PASS: staged serving_runtime package imports correctly")

# COMMAND ----------

# CELL 7 — Stable serving signature
input_example = pd.DataFrame([{
    "question": "Show India's GDP growth from 2015 to 2025.",
    "conversation_context": "",
}])
output_example = pd.DataFrame([{
    "status": "success", "route": "structured", "answer": "example",
    "total_latency_ms": 0.0, "plan_json": "{}",
    "citation_validation_json": "{}", "structured_json": "{}",
    "research_json": "{}",
}])
signature = infer_signature(input_example, output_example)
print(signature)

# COMMAND ----------

# CELL 8 — Log V3 candidate
mlflow.set_registry_uri("databricks-uc")
MODEL_FILE = str(PACKAGE_DIR / "worldbank_model.py")
PIP_REQUIREMENTS = [
    "mlflow>=3.12.0", "pandas", "pydantic",
    "databricks-sdk>=0.102.0", "databricks-openai", "databricks-ai-search",
]

# Automatic authentication passthrough: Model Serving grants the endpoint
# short-lived credentials scoped to exactly these resources. Without this
# list the serving container has no Databricks credentials, so every
# DatabricksOpenAI / AI Search / SQL call fails.
RESOURCES = [
    DatabricksServingEndpoint(endpoint_name=LLM_ENDPOINT),
    DatabricksServingEndpoint(endpoint_name=EMBEDDING_ENDPOINT),
    DatabricksVectorSearchIndex(index_name=AI_SEARCH_INDEX),
    DatabricksSQLWarehouse(warehouse_id=SQL_WAREHOUSE_ID),
    DatabricksTable(table_name=MACRO_TABLE),
]

with mlflow.start_run(run_name="worldbank-gep-production-v3") as run:
    model_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=MODEL_FILE,
        # Passing PACKAGE_DIR makes MLflow create code/serving_runtime/...
        code_paths=[str(PACKAGE_DIR)],
        signature=signature,
        input_example=input_example,
        pip_requirements=PIP_REQUIREMENTS,
        resources=RESOURCES,
    )
    run_id = run.info.run_id
print("Run ID:", run_id)
print("Logged model URI:", model_info.model_uri)

# COMMAND ----------

# CELL 9 — Isolated MLflow artifact load test BEFORE registration
# Remove the staged package path and cached modules so this test cannot
# accidentally succeed because /tmp/worldbank_model_package_v3 is on sys.path.
while package_root_str in sys.path:
    sys.path.remove(package_root_str)
for name in list(sys.modules):
    if name == "serving_runtime" or name.startswith("serving_runtime."):
        del sys.modules[name]
importlib.invalidate_caches()

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)
print("PASS: isolated MLflow artifact loaded successfully")

# Verify the packaged code, not the staging directory, supplied serving_runtime.
import serving_runtime as packaged_runtime
print("MLflow serving_runtime loaded from:", packaged_runtime.__file__)
if str(PACKAGE_ROOT) in str(packaged_runtime.__file__):
    raise RuntimeError("Isolation failed: serving_runtime came from staging directory")
print("PASS: serving_runtime resolved from MLflow artifact code path")

# COMMAND ----------

# CELL 10 — Register only the tested artifact
registered_model = mlflow.register_model(
    model_uri=model_info.model_uri,
    name=MODEL_NAME,
)
MODEL_VERSION = str(registered_model.version)
print("\n==========================================")
print("MODEL REGISTERED")
print("==========================================")
print("Model:", MODEL_NAME)
print("Version:", MODEL_VERSION)
print("Model URI:", model_info.model_uri)
print("\nNext: run 12_deploy_serving_endpoint_v3 (it deploys the latest version automatically).")
