# ============================================================
# World Bank GEP Intelligence Agent — Databricks App (Streamlit)
# ============================================================
#
# Chat UI in front of the Model Serving endpoint deployed by
# 06_production/12_deploy_serving_endpoint_v3.
#
# The endpoint name comes from the app resource "serving-endpoint"
# (see app.yaml). Authentication uses the app's service principal,
# which Databricks injects automatically — no token in code.
# ============================================================

import json
import os
import time
from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd
import streamlit as st
from databricks.sdk import WorkspaceClient
from databricks.sdk.config import Config


ENDPOINT_NAME = os.getenv("SERVING_ENDPOINT", "worldbank-gep-intelligence-agent")

# The endpoint scales to zero; the first request after idle can take
# several minutes while the container starts.
QUERY_TIMEOUT_SECONDS = 600

# How many previous turns to send as conversation_context for follow-ups.
CONTEXT_TURNS = 3
MAX_CONTEXT_CHARS = 3000

# Validated categorical palette, assigned in fixed order by entity.
SERIES_COLORS = [
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948",
]

ROUTE_LABELS = {
    "structured": "Historical data",
    "rag": "GEP research",
    "temporal_rag": "GEP comparison across editions",
    "hybrid": "Data + GEP research",
    "unknown": "Unsupported",
}

EXAMPLE_QUESTIONS = [
    "Show India's GDP growth from 2015 to 2025.",
    "What risks did the January 2025 Global Economic Prospects report highlight?",
    "How did downside risks change between the January 2022 and January 2025 GEP reports?",
    "Compare GDP growth in South Asia and Sub-Saharan Africa since 2022 and explain the January 2025 outlook.",
]


# ============================================================
# Endpoint client
# ============================================================

@st.cache_resource
def get_client() -> WorkspaceClient:
    # http_timeout_seconds is a Config attribute, not a WorkspaceClient kwarg.
    return WorkspaceClient(config=Config(http_timeout_seconds=QUERY_TIMEOUT_SECONDS))


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except ValueError:
        return None


def build_conversation_context(history: List[Dict[str, Any]]) -> str:
    """Summarize the last few turns so the Supervisor can resolve follow-ups."""
    lines = []
    for turn in history[-CONTEXT_TURNS:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['result'].get('answer', '')}")
    return "\n".join(lines)[-MAX_CONTEXT_CHARS:]


def ask_agent(question: str, conversation_context: str) -> Dict[str, Any]:
    started = time.perf_counter()
    response = get_client().serving_endpoints.query(
        name=ENDPOINT_NAME,
        dataframe_records=[{
            "question": question,
            "conversation_context": conversation_context,
        }],
    )
    predictions = response.as_dict().get("predictions") or []
    if not predictions:
        raise RuntimeError("The endpoint returned no prediction.")

    result = dict(predictions[0])
    for key in ["plan_json", "citation_validation_json", "structured_json", "research_json"]:
        result[key.removesuffix("_json")] = _parse_json(result.get(key))
    result["round_trip_ms"] = round((time.perf_counter() - started) * 1000)
    return result


# ============================================================
# Rendering
# ============================================================

def structured_frame(structured: Optional[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for series in (structured or {}).get("series") or []:
        entity = (series.get("entity") or {}).get("entity_name") or series.get("entity_name")
        for obs in series.get("observations") or []:
            rows.append({
                "Entity": entity,
                "Indicator": series.get("indicator_name") or series.get("indicator_code"),
                "Year": obs.get("year"),
                "Value": obs.get("value"),
                "Unit": obs.get("unit_label"),
            })
    return pd.DataFrame(rows)


def render_structured(structured: Optional[Dict[str, Any]]) -> None:
    df = structured_frame(structured)
    if df.empty:
        return

    entities = list(dict.fromkeys(df["Entity"]))
    charted = entities[: len(SERIES_COLORS)]
    color_scale = alt.Scale(domain=charted, range=SERIES_COLORS[: len(charted)])

    # One chart per indicator: indicators have different units and
    # must never share a y-axis.
    for indicator, group in df[df["Entity"].isin(charted)].groupby("Indicator", sort=False):
        unit = next((u for u in group["Unit"] if u), "")
        base = alt.Chart(group.dropna(subset=["Value"])).encode(
            x=alt.X("Year:O", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("Value:Q", title=unit or None),
            color=alt.Color(
                "Entity:N",
                scale=color_scale,
                legend=alt.Legend(orient="top", title=None) if len(charted) > 1 else None,
            ),
            tooltip=[
                alt.Tooltip("Entity:N"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Value:Q", format=",.2f"),
            ],
        )
        chart = (
            base.mark_line(strokeWidth=2)
            + base.mark_point(filled=True, size=64)
        ).properties(title=indicator, height=260)
        st.altair_chart(chart, width="stretch")

    if len(entities) > len(charted):
        st.caption(f"Chart shows the first {len(charted)} entities; the table lists all.")

    with st.expander("Data table"):
        st.dataframe(df, hide_index=True, width="stretch")


def render_sources(result: Dict[str, Any]) -> None:
    research = result.get("research") or {}
    evidence = research.get("evidence") or []
    if not evidence:
        return

    cited = set((result.get("citation_validation") or {}).get("cited_evidence_ids") or [])
    shown = [item for item in evidence if not cited or item.get("evidence_id") in cited]

    with st.expander(f"Sources ({len(shown)} GEP excerpts)"):
        for item in shown:
            location = ", ".join(
                str(part) for part in [item.get("chapter"), item.get("section")] if part
            )
            st.markdown(
                f"**[{item.get('evidence_id')}]** GEP January {item.get('report_year')}"
                f" · pages {item.get('page_start')}–{item.get('page_end')}"
                + (f" · {location}" if location else "")
            )
            st.caption(item.get("text") or "")


def render_result(result: Dict[str, Any]) -> None:
    status = result.get("status")
    answer = result.get("answer") or ""

    if status == "error":
        st.error(answer)
    elif status == "unsupported":
        st.info(answer)
    else:
        st.markdown(answer)

    render_structured(result.get("structured"))
    render_sources(result)

    route = result.get("route", "unknown")
    st.caption(
        f"Route: {ROUTE_LABELS.get(route, route)}"
        f" · agent {result.get('total_latency_ms', 0) / 1000:.1f}s"
        f" · round trip {result.get('round_trip_ms', 0) / 1000:.1f}s"
    )
    if result.get("plan"):
        with st.expander("Execution plan"):
            st.json(result["plan"])


# ============================================================
# Page
# ============================================================

st.set_page_config(page_title="GEP Intelligence Agent", page_icon="🌍", layout="centered")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("GEP Intelligence Agent")
    st.write(
        "Ask about historical World Bank indicators (2010–2025) or the January "
        "Global Economic Prospects reports (2022–2026)."
    )
    st.caption(f"Endpoint: `{ENDPOINT_NAME}`")
    if st.button("New conversation", width="stretch"):
        st.session_state.history = []
        st.rerun()

st.title("World Bank GEP Intelligence Agent")

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        render_result(turn["result"])

pending = None
if not st.session_state.history:
    st.write("Try one of these:")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, width="stretch"):
            pending = example

typed = st.chat_input("Ask about growth, inflation, debt, risks, outlook…")
question = typed or pending

if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking… the first request after idle can take a few minutes while the endpoint wakes up."):
            try:
                result = ask_agent(question, build_conversation_context(st.session_state.history))
            except Exception as exc:
                result = {
                    "status": "error",
                    "route": "unknown",
                    "answer": f"Could not reach the serving endpoint `{ENDPOINT_NAME}`: {exc}",
                }
        render_result(result)
    st.session_state.history.append({"question": question, "result": result})
