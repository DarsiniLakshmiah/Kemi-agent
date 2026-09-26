# Databricks notebook source
# MAGIC %md
# MAGIC # 12_deploy_serving_endpoint_v3
# MAGIC
# MAGIC Deploy the latest registered version of the World Bank GEP model to Model Serving. Run this only after Notebook 11 V3 finishes.
# MAGIC
# MAGIC - Deploys the **latest** Unity Catalog model version by default (set `MODEL_VERSION_OVERRIDE` to pin one).
# MAGIC - Creates the endpoint if it doesn't exist yet (e.g. a fresh workspace), otherwise updates it in place.
# MAGIC - If deployment fails, prints the container build and server logs so the real error is visible.

# COMMAND ----------

# MAGIC %pip install -q "databricks-sdk>=0.102.0"

# COMMAND ----------

# Restart Python so the SDK installed above is the one imported below.
dbutils.library.restartPython()

# COMMAND ----------

# CELL 2 — Configuration
from datetime import timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    ServingModelWorkloadType,
)

MODEL_NAME = "worldbank_ai.ai.gep_intelligence_agent"
ENDPOINT_NAME = "worldbank-gep-intelligence-agent"
SQL_WAREHOUSE_ID = "9b3ae64f1848fe3f"

# Leave as None to deploy the newest registered version.
MODEL_VERSION_OVERRIDE = None

# Agent containers routinely take longer than the SDK's 20-minute default.
DEPLOY_TIMEOUT = timedelta(minutes=45)

w = WorkspaceClient()

if MODEL_VERSION_OVERRIDE is not None:
    MODEL_VERSION = str(MODEL_VERSION_OVERRIDE)
else:
    versions = [int(v.version) for v in w.model_versions.list(full_name=MODEL_NAME)]
    if not versions:
        raise RuntimeError(f"No registered versions of {MODEL_NAME}. Run Notebook 11 first.")
    MODEL_VERSION = str(max(versions))

SERVED_ENTITY_NAME = f"gep-intelligence-agent-v{MODEL_VERSION}"

print("Model:", MODEL_NAME)
print("Version:", MODEL_VERSION)
print("Endpoint:", ENDPOINT_NAME)
print("Served entity:", SERVED_ENTITY_NAME)

# COMMAND ----------

# CELL 3 — Check whether the endpoint already exists
try:
    endpoint = w.serving_endpoints.get(name=ENDPOINT_NAME)
    ENDPOINT_EXISTS = True
    print("Endpoint ID:", endpoint.id)
    print("Current state:", endpoint.state)
    if endpoint.config:
        print("Active config version:", endpoint.config.config_version)
    if endpoint.pending_config:
        print("Pending config version:", endpoint.pending_config.config_version)
except NotFound:
    ENDPOINT_EXISTS = False
    print(f"Endpoint '{ENDPOINT_NAME}' does not exist yet; it will be created.")

# COMMAND ----------

# CELL 4 — Define the served entity
served_entity = ServedEntityInput(
    name=SERVED_ENTITY_NAME,
    entity_name=MODEL_NAME,
    entity_version=MODEL_VERSION,
    workload_type=ServingModelWorkloadType.CPU,
    workload_size="Small",
    scale_to_zero_enabled=True,
    environment_vars={
        "DATABRICKS_SQL_WAREHOUSE_ID": SQL_WAREHOUSE_ID,
    },
)
print("Prepared served entity:", served_entity.name)

# COMMAND ----------

# CELL 5 — Create or update the endpoint, printing logs on failure
def print_deployment_logs():
    """Show why the serving container failed; the SDK exception alone doesn't."""
    for label, fetch in [
        ("BUILD LOGS", w.serving_endpoints.build_logs),
        ("SERVER LOGS", w.serving_endpoints.logs),
    ]:
        print("\n" + "=" * 70)
        print(label)
        print("=" * 70)
        try:
            print(fetch(name=ENDPOINT_NAME, served_model_name=SERVED_ENTITY_NAME).logs)
        except Exception as log_exc:
            print(f"Could not fetch {label.lower()}: {type(log_exc).__name__}: {log_exc}")


try:
    if ENDPOINT_EXISTS:
        print("Updating existing endpoint to model version", MODEL_VERSION)
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME,
            served_entities=[served_entity],
            timeout=DEPLOY_TIMEOUT,
        )
    else:
        print("Creating endpoint with model version", MODEL_VERSION)
        w.serving_endpoints.create_and_wait(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(
                name=ENDPOINT_NAME,
                served_entities=[served_entity],
            ),
            timeout=DEPLOY_TIMEOUT,
        )
    print("Deployment completed")
except Exception as exc:
    print(f"DEPLOYMENT FAILED: {type(exc).__name__}: {exc}")
    print_deployment_logs()
    raise

# COMMAND ----------

# CELL 6 — Final READY validation
endpoint = w.serving_endpoints.get(name=ENDPOINT_NAME)
print("Final state:", endpoint.state)
print("Active config version:", endpoint.config.config_version if endpoint.config else None)

served_versions = []
if endpoint.config:
    for entity in endpoint.config.served_entities or []:
        print("Served entity:", entity.name, "| model:", entity.entity_name, "| version:", entity.entity_version)
        served_versions.append(str(entity.entity_version))

state_text = str(endpoint.state)
if "READY" not in state_text or "UPDATE_FAILED" in state_text:
    raise RuntimeError(f"Endpoint is not READY: {endpoint.state}")
if MODEL_VERSION not in served_versions:
    raise RuntimeError(f"Endpoint is READY but serving versions {served_versions}, not {MODEL_VERSION}")
print("PASS: endpoint is READY on model version", MODEL_VERSION)
print("Next: 13_endpoint_smoke_tests")
