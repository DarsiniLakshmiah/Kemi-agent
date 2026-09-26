# global-economic-prospects-agent

A Databricks agent that answers questions about World Bank macroeconomic indicators (2010–2025) and the January *Global Economic Prospects* (GEP) reports (2022–2026). It is deployed as an MLflow model on Model Serving, with a Streamlit Databricks App as the chat UI.

Example questions:

- *Show India's GDP growth from 2015 to 2025.* → historical indicator data
- *What risks did the January 2025 GEP highlight for South Asia?* → report research
- *How did downside risks change between the 2022 and 2025 GEP reports?* → comparison across report editions
- *Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 and explain the 2025 outlook.* → data and research combined

## How it works

```
question ──► input guardrails ──► Supervisor ──┬─► Data Agent (SQL) ──────────┐
                                  (routes)     │                              ├─► Synthesis Agent ──► output guardrails ──► cited answer
                                               └─► Research Agent (RAG) ──────┘
```

- **Supervisor** turns the question into a validated plan. The route is one of:
  - `structured`: historical indicator data only
  - `rag`: evidence from one GEP report edition
  - `temporal_rag`: a comparison across GEP report editions
  - `hybrid`: indicator data plus report evidence
  - `unknown`: the question is unsupported or too unclear to route

  The plan keeps two kinds of year apart: indicator observation years (2010–2025) and GEP report editions (2022–2026). It also resolves follow-ups such as "What about 2026?" from the conversation context.
- **Data Agent** runs governed SQL against `worldbank_ai.gold.macroeconomic_indicators` on a SQL warehouse.
- **Research Agent** runs a hybrid (keyword and vector) search over the GEP chunks in the `worldbank_ai.rag.gep_structure_v1_qwen3_index` AI Search index, filtered by report year. It then drafts an answer that cites the evidence it retrieved.
- **Synthesis Agent** merges the two results into the final answer.
- **Guardrails** check the input for prompt injection before anything runs. Afterwards they check that every citation points to evidence that was actually retrieved.

Models:
- LLM: `databricks-meta-llama-3-3-70b-instruct`
- Embeddings: `databricks-qwen3-embedding-0-6b`

## Before you start

- A Unity Catalog workspace where you can create the `worldbank_ai` catalog. Databricks Free Edition works.
- A SQL warehouse. Put its ID in `SQL_WAREHOUSE_ID` in `06_production/10`, `11_mlflow_package_register_v3` and `12_deploy_serving_endpoint_v3`.
- Access to the `databricks-meta-llama-3-3-70b-instruct` and `databricks-qwen3-embedding-0-6b` serving endpoints.
- The five GEP PDFs, named `GEP-Jan-2022.pdf` … `GEP-Jan-2026.pdf`. Upload them after `00_setup/02` creates the volume, to:

  `/Volumes/worldbank_ai/bronze/raw_documents/global_economic_prospects/`

  Later notebooks check page counts against these exact editions (1,098 pages in total).

Import this repo as a Databricks **Git folder** (**Workspace → Create → Git folder**) and run the notebooks from there. Paths are resolved relative to the notebooks, so the folder name doesn't matter.

If you upload notebooks by hand instead, use **Import → File**. Don't paste an `.ipynb` file's contents into a cell: it is JSON, and running it fails with `NameError: name 'false' is not defined`.

## Run order

Run each notebook top to bottom.

| Step | Notebooks | Creates |
|---|---|---|
| Setup | `00_setup/00` → `01` → `02` (upload PDFs, then rerun `02`) → `03` | catalog, schemas, volume |
| Ingestion | `01_ingestion/structured/01` → `02` → `03` | bronze tables (World Bank API) |
| Silver | `02_Silver/structured/01` → `02` → `03`, then `02_Silver/unstructured/01` → `02` → `03` → `04` | cleaned tables, GEP chunks |
| Gold | `03_gold/structured/01` → `02` | `gold.macroeconomic_indicators` |
| Search index | `04_rag/…/01_chunking_experiments` → `03_prepare_retrieval_corpora` → `04_ai_search_indexes_qwen3` | AI Search endpoint + indexes |
| Deploy | `06_production/09` → `10` → `11_mlflow_package_register_v3` → `12_deploy_serving_endpoint_v3` → `13_endpoint_smoke_tests` | registered model + serving endpoint |
| App | `07_app` (see below) | chat UI |

Notes:

- **Search index.** The first run takes a while, because the AI Search endpoint and then each index provision before the initial sync:
  - Seeing `PROVISIONING_ENDPOINT` and `sync()` returning `BadRequest: ... is not ready` during that time is normal.
  - Wait until each index reports `ready=True` and its `indexed_row_count` matches the chunk table.
  - The notebook only creates what doesn't exist yet, so it is safe to rerun.
- **11 and 12 exist as both `.py` and `.ipynb`, with the same code.** Use either one.
  - `11` registers a new model version.
  - `12` deploys the newest version, creating the endpoint on the first run and updating it afterwards. The first deployment can take up to about 45 minutes.
- **Re-run `11` → `12` → `13` after changing anything in `serving_runtime/`.** The served model only contains the code that existed when `11` packaged it.

### Optional: evaluation and experiments

These measure quality and don't affect deployment:

- `04_rag/…/02, 05–11`:
  - `02` builds the 50-question benchmark with an LLM.
  - `05`, `06` and `11` require that benchmark.
- `05_tools_agents/*`:
  - Development versions of the agents, each with routing and behaviour tests.
  - `08_agent_evaluation_V4` runs the end-to-end benchmark.

The benchmark notebooks assert exact counts, for example 10 approved questions per report year. LLM annotations can vary between runs, so rerun `02` if one year comes up short.

## Deploy the app

1. **Compute → Apps → Create app → Custom app**. Link this GitHub repo in the Git step if you want to deploy from Git.
2. Under **App resources**, add a **Serving endpoint** resource:
   - endpoint: `worldbank-gep-intelligence-agent`
   - permission: **Can query**
   - resource key: `serving-endpoint`. It must be exactly this, because `07_app/app.yaml` reads the endpoint name from it.
3. Click **Create app**, then **Deploy**:
   - from Git: branch `main`, source code path `07_app`
   - from the workspace: the full path of the `07_app` folder

   Deploying from Git reads the repo as it is on GitHub, so push your changes before redeploying.

The app is billed while it's running, so stop it when you're not using it.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `sync()` returns `BadRequest: Vector index ... is not ready` | The index is still provisioning. Wait; the first sync runs automatically. |
| `NameError: name 'false' is not defined` | An `.ipynb` file was pasted into a cell. Import the file instead. |
| `FileNotFoundError: Could not find serving_runtime/worldbank_model.py` in `11` | `serving_runtime/` must sit next to `06_production/` in the workspace. |
| `12` fails while deploying | It prints the endpoint's build and server logs. The cause is usually in the server log. |
| The app says "Could not reach the serving endpoint" | Check the app resource key is `serving-endpoint` and the endpoint is `READY`. The app's **Logs** tab shows the full error. |
| `14_monitoring_audit` returns no rows | Expected: nothing writes to `monitoring.application_logs` yet (see below). |

## Monitoring

`06_production/14_monitoring_audit` queries `worldbank_ai.monitoring.application_logs` and `user_feedback`. Notebook `09` creates these tables, but nothing writes to them yet, so the queries return no rows. To fill them, either:
- turn on **inference tables** for the serving endpoint under **AI Gateway**, and point `14` at that table, or
- have the app write a row per request and per feedback click.

## Costs

These bill while they're running:
- **The Databricks App:** stop it when idle.
- **The AI Search endpoint:** delete it when you're done experimenting.
- **The serving endpoint:** it scales to zero when idle.

## Layout

- `00_setup/` … `03_gold/`: the data pipeline, from bronze to silver to gold.
- `04_rag/`: chunking, retrieval corpora, AI Search indexes and the RAG evaluation experiments.
- `05_tools_agents/`: development notebooks for each agent.
  - `05_tools_agents/runtime/` holds notebook versions of the agents, used with `%run`.
- `serving_runtime/`: plain Python modules packaged into the served model.
  - They are not notebooks, so don't add the `# Databricks notebook source` header to them.
  - `supervisor_runtime.py` has the same prompt as `05_tools_agents/03_supervisor_agent.py` and `05_tools_agents/runtime/supervisor_runtime.py`. Keep all three in sync.
- `06_production/`: packaging, deployment, smoke tests and monitoring.
  - `06_production/_archive/` holds earlier attempts, kept for reference; use the `_v3` notebooks.
- `07_app/`: the Streamlit chat app (`app.py`, `app.yaml`, `requirements.txt`).
