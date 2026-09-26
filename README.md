# global-economic-prospects-agent

A Databricks agent that answers questions about World Bank macroeconomic indicators (2010–2025) and the January *Global Economic Prospects* (GEP) reports (2022–2026). A supervisor routes each question to a SQL data agent, a RAG research agent over the GEP reports, or both, then a synthesis agent writes a cited answer. It is deployed as an MLflow model on Model Serving with a Streamlit Databricks App in front.

## Before you start

- A Unity Catalog workspace where you can create the `worldbank_ai` catalog.
- A SQL warehouse. Put its ID in `SQL_WAREHOUSE_ID` in `06_production/10`, `11_…_v3`, and `12_…_v3`.
- Access to the `databricks-meta-llama-3-3-70b-instruct` and `databricks-qwen3-embedding-0-6b` serving endpoints.
- The five GEP PDFs, uploaded after `00_setup/02` creates the volume, to
  `/Volumes/worldbank_ai/bronze/raw_documents/global_economic_prospects/`
  named `GEP-Jan-2022.pdf` … `GEP-Jan-2026.pdf`.
  Later notebooks check page counts against these exact editions (1,098 pages in total).

Import this repo as a Databricks **Git folder** and run the notebooks from there. Paths are resolved relative to the notebooks, so the folder name doesn't matter.

## Run order

Run each notebook top to bottom.

| Step | Notebooks | Creates |
|---|---|---|
| Setup | `00_setup/00` → `01` → `02` (upload PDFs, then rerun `02`) → `03` | catalog, schemas, volume |
| Ingestion | `01_ingestion/structured/01` → `02` → `03` | bronze tables (World Bank API) |
| Silver | `02_Silver/structured/01` → `02` → `03`, then `02_Silver/unstructured/01` → `02` → `03` → `04` | cleaned tables, GEP chunks |
| Gold | `03_gold/structured/01` → `02` | `gold.macroeconomic_indicators` |
| Search index | `04_rag/…/01_chunking_experiments` → `03_prepare_retrieval_corpora` → `04_ai_search_indexes_qwen3` | AI Search endpoint + indexes (takes a while) |
| Deploy | `06_production/09` → `10` → `11_mlflow_package_register_v3` → `12_deploy_serving_endpoint_v3` → `13_endpoint_smoke_tests` | registered model + serving endpoint |
| App | `07_app` (see below) | chat UI |

`04_ai_search_indexes_qwen3` only creates the endpoint and indexes that don't exist yet, so it is safe to rerun.

### Optional: evaluation and experiments

These measure quality and don't affect deployment:

- `04_rag/…/02, 05–11`. `02` builds the 50-question benchmark with an LLM; `05`, `06` and `11` require it.
- `05_tools_agents/*`. Development versions of the agents; `08_agent_evaluation_V4` runs the end-to-end benchmark.

The benchmark notebooks assert exact counts (for example 10 approved questions per report year). LLM annotations can vary between runs, so rerun `02` if one year comes up short.

## Deploy the app

1. **Compute → Apps → Create app → Custom app**.
2. Under **App resources**, add a **Serving endpoint** resource:
   - endpoint: `worldbank-gep-intelligence-agent`
   - permission: **Can query**
   - resource key: `serving-endpoint`
3. **Deploy** with the source code path set to this repo's `07_app` folder.

The app is billed while it's running, so stop it when you're not using it.

## Layout

- `serving_runtime/`: plain Python modules packaged into the served model. They are not notebooks, so don't add the `# Databricks notebook source` header to them.
- `05_tools_agents/runtime/`: notebook versions of the same agents, used with `%run` during development.
- `06_production/_archive/`: earlier packaging and deploy attempts, kept for reference; use the `_v3` notebooks instead.
- `06_production/14_monitoring_audit`: queries `monitoring.application_logs`. Nothing writes to that table yet, so its results stay empty until request logging is added.
