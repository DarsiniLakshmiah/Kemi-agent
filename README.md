# Global Economic Prospects Intelligence Agent

A governed AI engineering project on Databricks. It combines World Bank macroeconomic indicators with retrieval-augmented generation (RAG) over the World Bank's *Global Economic Prospects* (GEP) reports. Each question is routed to a SQL data agent, a citation-grounded research agent, or both, and the answer comes back through a chat app.

## Project status

The whole system is built and deployed end to end:
- the data platform
- the RAG pipeline
- the multi-agent orchestration
- the evaluation benchmarks and guardrails
- MLflow packaging and Unity Catalog registration
- a Model Serving endpoint
- a Streamlit Databricks App that queries the endpoint

The deployed app routes questions correctly. The most recent fixes, listed under [Recent fixes](#recent-fixes), are in the code but need one more package-and-deploy cycle, meaning `06_production/11` → `12` → `13`, before they are live. Request and feedback logging for monitoring is not wired up yet.

## Example questions

| Question | Route |
|---|---|
| Show India's GDP growth from 2015 to 2025. | `structured` |
| What risks did the January 2025 GEP report highlight? | `rag` |
| How did downside risks change between the January 2022 and January 2025 GEP reports? | `temporal_rag` |
| Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 and explain the January 2025 outlook. | `hybrid` |
| *(after a GEP 2025 question)* What about 2026? | follow-up, resolved to `rag`, 2026 edition |

## Architecture

### Data and retrieval pipeline

```text
World Bank Indicators API                     GEP PDFs (Jan 2022–2026)
          |                                             |
          v                                             v
   Bronze Delta tables                      Unity Catalog volume (bronze)
          |                                             |
          v                                             v
   Silver (cleaned, ISO3-canonical)          Parse -> Clean -> Chunk -> Metadata
          |                                             |
          v                                             v
   Gold: macroeconomic_indicators            Qwen3 embeddings -> Databricks AI Search
```

### Serving path

```text
Databricks App (Streamlit)
        |   question + last 3 turns as conversation context
        v
Model Serving endpoint: worldbank-gep-intelligence-agent
        |
Input guardrails (prompt-injection check)
        |
Supervisor  ── validated plan: route, entities, indicators,
        |      observation years, GEP report editions
   +----+-----------------------+
   v                            v
Data Agent                 Research Agent
(parameterized SQL)        (hybrid search + year filter,
   |                        top 6 evidence chunks, cited draft)
SQL Warehouse                   |
   |                        AI Search index
Gold Delta table                |
   +------------+---------------+
                v
         Synthesis Agent
                |
   Output guardrails (citation / source contract)
                |
             Response  ->  app renders answer, charts, tables, evidence
```

### Components

| Component | Implementation |
|---|---|
| LLM | `databricks-meta-llama-3-3-70b-instruct` (Foundation Model APIs) |
| Embeddings | `databricks-qwen3-embedding-0-6b` |
| Vector search | AI Search endpoint `worldbank-gep-ai-search`, index `worldbank_ai.rag.gep_structure_v1_qwen3_index` |
| Structured data | `worldbank_ai.gold.macroeconomic_indicators` via SQL Statement Execution on a SQL warehouse |
| Model | `worldbank_ai.ai.gep_intelligence_agent` (MLflow pyfunc, Unity Catalog) |
| Serving | Model Serving endpoint `worldbank-gep-intelligence-agent`, CPU Small, scale-to-zero |
| UI | Streamlit Databricks App (`07_app`) |

Technology stack: Python, SQL, PySpark, Delta Lake, Unity Catalog, Databricks SQL Warehouse, Databricks AI Search, Model Serving, MLflow, Foundation Model APIs, Streamlit and Altair.

## What we built

### 1. Lakehouse

```text
worldbank_ai
├── bronze       raw API pulls, raw GEP PDFs (volume)
├── silver       cleaned indicators, countries, parsed/cleaned/chunked documents
├── gold         macroeconomic_indicators (agent-facing)
├── rag          retrieval corpora and AI Search indexes
├── ai           registered model
└── monitoring   application_logs, user_feedback (tables created, not yet populated)
```

**Structured data.**
- 15 World Bank indicators:
  - growth: GDP growth, GDP per capita growth, GDP, GDP per capita
  - prices and trade: inflation, trade, exports, imports
  - investment and government: gross capital formation, government expense, revenue and debt
  - external and savings: savings, FDI, current account
- 63,600 observation rows across 265 entities.
- The requested window is 2010–2026, but historical observations stop at 2025. The system never invents 2026 observations.
- Missing values stay `null`; they are never replaced with zero.

**Data-quality fix.** Observations used ISO2 codes while the country dimension used ISO3, so joins silently dropped rows. The Silver pipeline now maps each source entity to its ISO2 code and converts it to ISO3 before the Gold step.

**Documents.**
- Five January GEP editions, 2022–2026, 1,098 pages in total.
- An early cleaning rule deleted numeric lines from a commodity table. It was removed, and a regression test now protects numeric and tabular content.

### 2. Chunking and retrieval

We compared three chunking strategies:
- fixed-size
- recursive
- structure-aware parent-child

Structure-aware parent-child won on measured retrieval quality. It produces 931 parent chunks and 2,215 child chunks.

Retrieval uses **hybrid search** (keyword and vector) with a **report-year filter** when the plan names editions. It returns the top 6 evidence chunks, labelled `E1`, `E2`, …, and generated answers must cite those labels.

**Retrieval benchmark.** We started from 80 candidate questions, 73 were usable, and we kept 50: 10 per GEP edition.

| Metric | Result |
|---|---|
| Hit@1 | 0.66 |
| Hit@3 | 0.84 |
| Hit@5 | 0.92 |
| Hit@6 | 0.94 |
| MRR@6 | 0.756 |

**Reranking experiment.** LLM reranking raised MRR from 0.756 to 0.788 and Hit@3 from 0.84 to 0.92. But it added about 9.7 s of latency, against about 0.2 s for retrieval alone. The served model therefore does not rerank.

### 3. RAG answer quality (50-question benchmark)

| Metric | Result |
|---|---|
| Citation validity | 1.00 |
| Answer citation rate | 1.00 |
| Unsupported claim rate | 0.00 |
| Gold evidence cited | 0.90 |
| Hallucination rate | 0.02 |
| Major omission rate | 0.06 |
| Groundedness | 4.94 / 5 |
| Answer relevance | 4.98 / 5 |
| Reference coverage | 4.92 / 5 |

MLflow tracks the experiments and evaluations. It is not a data layer.

### 4. Multi-agent system

- **Supervisor** combines deterministic extraction (indicator codes, explicit years, route hints) with LLM interpretation, then validates the plan in code:
  - Route and tool flags must agree.
  - Only registry indicators and loaded GEP editions survive validation.
  - Observation years (2010–2025) are kept apart from report editions (2022–2026).
  - Follow-up questions are resolved from the conversation context.
- **Data Agent** uses governed, parameterized SQL functions. There is no free-form text-to-SQL. It doesn't depend on Spark, so it runs inside the serving container.
- **Research Agent** runs hybrid retrieval with year filters, builds the context with de-duplication, drafts a grounded answer and validates its citations.
- **Synthesis Agent** merges historical facts (`S` citations) and GEP evidence (`E` citations) while keeping the two sources separate.
- **Guardrails:**
  - Before execution: a prompt-injection check and plan validation.
  - After execution: every cited evidence ID must be one that was actually retrieved.

### 5. Agent benchmark (40 cases, 10 per route)

| Metric | Result |
|---|---|
| Cases executed | 40 / 40 |
| Route accuracy | 1.00 |
| Report-year accuracy | 1.00 |
| Citation validation | 1.00 |
| Execution success | 1.00 |
| Average latency | ~7.1 s |

This is 100% on this 40-case benchmark, not a claim that the system is 100% accurate.

### 6. Productionization

- **Packaging.** `serving_runtime/` is plain Python modules, packaged under MLflow `code/serving_runtime/`. An isolated load test confirms the package loads from the MLflow artifact and not from the workspace.
- **Credentials.** Every Databricks resource the agent calls is declared with MLflow `resources=`, so Model Serving issues the endpoint short-lived, scoped credentials: the LLM and embedding endpoints, the AI Search index, the SQL warehouse and the Gold table.
- **Lazy clients.** Databricks clients are created on the first request, not at import. Creating them at import tried to authenticate while the model was loading and blocked an earlier deployment.
- **Deployment.** The deploy notebook picks the newest registered version and creates or updates the endpoint. If deployment fails, it prints the build and server logs.
- **App.** A Streamlit Databricks App calls the endpoint through an app resource. It renders the answer, charts and tables for indicator data, and the GEP evidence behind the answer.

## Recent fixes

These came out of the first end-to-end runs in a fresh workspace:

| Issue | Fix | Where |
|---|---|---|
| The first question after the SQL warehouse was idle failed with `StatementState.CANCELED`, because the query was cancelled after 30 s while the warehouse was still starting. | Keep the statement running after 30 s and poll for up to 180 s, then cancel with a clear message. | `serving_runtime/data_agent_sql_runtime.py` *(needs 11 → 12 → 13)* |
| Follow-ups like "What about 2026?" weren't resolved from context. | Added a follow-up rule to the Supervisor prompt. | `serving_runtime/supervisor_runtime.py` and dev copies *(needs 11 → 12 → 13)* |
| Creating the endpoint failed: `EndpointCoreConfigInput` missing `name`. | Pass the endpoint name, which newer `databricks-sdk` versions require. | `06_production/12_deploy_serving_endpoint_v3` |
| The app failed with `WorkspaceClient got an unexpected keyword argument 'http_timeout_seconds'`. | Set the timeout through `Config` instead. | `07_app/app.py` |
| Notebook 11 would register the model twice. | Removed the duplicate registration cell. | `06_production/11_mlflow_package_register_v3` |

## Limitations

**Data coverage**
- Only the five **January** GEP editions (2022–2026) are indexed. June editions and other World Bank publications are not.
- Historical indicators cover **15 indicators, 2010–2025**. Questions about other indicators are declined rather than approximated.
- There is **no forecast tool**. Forecasts come only from GEP report text, never from the structured data.

**Quality**
- The benchmarks are small: 50 RAG questions and 40 agent cases. The questions were generated and annotated with an LLM, and several metrics are LLM-judged. The scores show the system works on these sets, not general accuracy.
- The RAG evaluation still measured a 2% hallucination rate and a 6% major-omission rate.
- **Reranking is off** in production, which trades some ranking quality (Hit@3 0.84 against 0.92) for latency.
- Routing depends on the LLM following the Supervisor prompt. Code validation catches malformed plans but not every wrong routing decision.
- Follow-up handling uses the last 3 turns as text. There is no persistent memory or entity tracking across a longer conversation.

**Safety**
- Input guardrails are **pattern-based** prompt-injection checks. There is no ML classifier, PII detection or toxicity filter.
- Output guardrails check citations against retrieved evidence. They cannot prove that each sentence is supported by the text it cites.

**Operations**
- **Cold starts.** The serving endpoint scales to zero and the SQL warehouse auto-stops, so the first request after an idle period can take much longer than the ~7 s benchmark average.
- **No monitoring data yet.** `monitoring.application_logs` and `user_feedback` exist but nothing writes to them, so `14_monitoring_audit` returns no rows.
- The **Supervisor prompt exists in three files**: the serving runtime, the dev notebook and the dev runtime. They have to be kept in sync by hand.
- Single workspace, no CI/CD. Deployment is done by running notebooks by hand.

## Roadmap

1. Package and deploy the recent fixes (`11` → `12` → `13`), then re-run the agent benchmark against the endpoint.
2. Log requests and feedback, through inference tables or app writes, and point `14_monitoring_audit` at them.
3. Add June GEP editions and grow the benchmarks with human-reviewed questions.
4. Try conditional reranking only for low-confidence retrievals.
5. Add stronger guardrails (PII, toxicity) and CI for packaging and deployment.

## How to run it

### Prerequisites

- A Unity Catalog workspace where you can create the `worldbank_ai` catalog. Databricks Free Edition works.
- A SQL warehouse. Put its ID in `SQL_WAREHOUSE_ID` in `06_production/10`, `11_mlflow_package_register_v3` and `12_deploy_serving_endpoint_v3`.
- Access to the `databricks-meta-llama-3-3-70b-instruct` and `databricks-qwen3-embedding-0-6b` endpoints.
- The five GEP PDFs, named `GEP-Jan-2022.pdf` … `GEP-Jan-2026.pdf`. Upload them after `00_setup/02` creates the volume, to `/Volumes/worldbank_ai/bronze/raw_documents/global_economic_prospects/`.

Import the repo as a Databricks **Git folder** (**Workspace → Create → Git folder**). If you upload files by hand, use **Import → File**. Pasting an `.ipynb` file's contents into a cell fails with `NameError: name 'false' is not defined`.

### Run order

Run each notebook top to bottom.

| Step | Notebooks | Creates |
|---|---|---|
| Setup | `00_setup/00` → `01` → `02` (upload PDFs, then rerun `02`) → `03` | catalog, schemas, volume |
| Ingestion | `01_ingestion/structured/01` → `02` → `03` | bronze tables |
| Silver | `02_Silver/structured/01` → `02` → `03`, then `02_Silver/unstructured/01` → `02` → `03` → `04` | cleaned tables, GEP chunks |
| Gold | `03_gold/structured/01` → `02` | `gold.macroeconomic_indicators` |
| Search index | `04_rag/…/01_chunking_experiments` → `03_prepare_retrieval_corpora` → `04_ai_search_indexes_qwen3` | AI Search endpoint + indexes |
| Deploy | `06_production/09` → `10` → `11_…_v3` → `12_…_v3` → `13_endpoint_smoke_tests` | model version + serving endpoint |
| App | `07_app` (below) | chat UI |

Notes:
- **Search index.** On a first run the AI Search endpoint and then each index provision before the initial sync, which takes a while.
  - `PROVISIONING_ENDPOINT` and `sync()` returning `BadRequest: ... is not ready` are normal during that time.
  - The notebook only creates what doesn't exist yet, so it is safe to rerun.
- **Changes to `serving_runtime/`** only reach production after `11` → `12` → `13`.
- **11 and 12 exist as both `.py` and `.ipynb`,** with the same code.
- **Optional evaluation:**
  - `04_rag/…/02, 05–11`: the RAG benchmarks. `02` builds the question set; `05`, `06` and `11` need it.
  - `05_tools_agents/08_agent_evaluation_V4`: the agent benchmark.

### Deploy the app

1. **Compute → Apps → Create app → Custom app**.
2. **App resources → Add resource → Serving endpoint**:
   - endpoint: `worldbank-gep-intelligence-agent`
   - permission: **Can query**
   - resource key: `serving-endpoint`. It must be exactly this, because `07_app/app.yaml` reads it.
3. **Create app**, then **Deploy**:
   - from Git: branch `main`, source path `07_app`. Push your changes first.
   - from the workspace: the full path of the `07_app` folder

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `sync()` returns `BadRequest: Vector index ... is not ready` | The index is still provisioning. The first sync runs automatically. |
| `FileNotFoundError: ... serving_runtime/worldbank_model.py` in `11` | `serving_runtime/` must sit next to `06_production/`. |
| `12` fails | It prints the build and server logs. The cause is usually in the server log. |
| The app says "Could not reach the serving endpoint" | Check the resource key is `serving-endpoint` and the endpoint is `READY`. See the app's **Logs** tab. |
| `SQL statement failed: StatementState.CANCELED` | The warehouse was starting up. Fixed in the current code; until it's redeployed, start the warehouse first. |
| `14_monitoring_audit` is empty | Expected; see [Limitations](#limitations). |

### Costs

- **The Databricks App** bills while running, so stop it when idle.
- **The AI Search endpoint** bills while it exists, so delete it when you're done.
- **The serving endpoint** scales to zero when idle.

## Repository layout

```text
00_setup/            catalog, schemas, volume, config
01_ingestion/        World Bank API -> bronze
02_Silver/           structured cleaning; PDF parse/clean/chunk/metadata
03_gold/             agent-facing Gold tables
04_rag/              chunking experiments, corpora, AI Search indexes, RAG evaluation
05_tools_agents/     development notebooks per agent (+ runtime/ for %run)
serving_runtime/     plain Python package served by MLflow (not notebooks)
06_production/       config tables, packaging, deployment, smoke tests, monitoring
07_app/              Streamlit Databricks App
```

## Design principles

1. Keep historical facts and forecasts separate.
2. Keep missing numeric values as null.
3. Answer structured questions with deterministic, governed tools, not free-form SQL.
4. Ground every qualitative claim in retrieved GEP evidence, with explicit citations.
5. Evaluate retrieval and generation separately.
6. Pay for expensive steps such as reranking only when measured gains justify the latency.
7. Keep development notebooks separate from serving code, and don't assume Spark exists in the serving container.
8. Make no import-time network calls in served code.
9. Claim a component is production-ready only after it passes validation.
