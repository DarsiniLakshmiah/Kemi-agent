# ============================================================
# World Bank GEP Intelligence Agent
# Production Orchestrator
# ============================================================

import time
import traceback
from typing import Any, Dict, List, Optional

from serving_runtime.supervisor_runtime import (
    run_supervisor_agent,
)

from serving_runtime.data_agent_sql_runtime import (
    run_data_agent,
)

from serving_runtime.research_agent_runtime import (
    run_research_agent,
)

from serving_runtime.synthesis_agent_runtime import (
    run_synthesis_agent,
)

from serving_runtime.guardrails_runtime import (
    validate_user_input,
    validate_execution_plan,
    apply_post_execution_guardrails,
)


UNKNOWN_ROUTE_ANSWER = (
    "I can answer questions about historical World Bank macroeconomic "
    "indicators (2010-2025) and the January Global Economic Prospects "
    "reports (2022-2026). I could not map this question to either source. "
    "Please rephrase it with a country/region, an indicator, or a GEP topic."
)


def _extract_allowed_evidence_ids(
    research_result: Optional[Dict[str, Any]],
) -> List[str]:
    """
    Evidence IDs the Research Agent actually supplied.

    run_research_agent() returns the evidence list directly under
    "evidence"; each item carries its [E#] id.
    """
    if not research_result:
        return []

    return [
        str(item["evidence_id"])
        for item in research_result.get("evidence") or []
        if isinstance(item, dict) and item.get("evidence_id")
    ]


def execute_agent(
    question: str,
    conversation_context: Optional[str] = None,
) -> Dict[str, Any]:
    started = time.perf_counter()

    plan = None
    route = "unknown"
    data_result = None
    research_result = None

    def _elapsed_ms() -> float:
        return round((time.perf_counter() - started) * 1000, 2)

    try:
        # Reject empty/oversized/prompt-injection input before any LLM call.
        input_check = validate_user_input(question)

        plan = run_supervisor_agent(
            question,
            conversation_context=conversation_context,
        )
        route = plan["route"]

        # "unknown" is a valid Supervisor outcome, not an error. The plan
        # guardrail only accepts executable routes, so answer here.
        if route == "unknown":
            return {
                "status": "unsupported",
                "question": question,
                "route": route,
                "plan": plan,
                "answer": UNKNOWN_ROUTE_ANSWER,
                "citation_validation": None,
                "data_agent_result": None,
                "research_agent_result": None,
                "guardrails": {"input": input_check},
                "total_latency_ms": _elapsed_ms(),
            }

        # Validates the plan; the full Supervisor plan (with entities,
        # indicators and years) is what the downstream agents consume.
        plan_check = validate_execution_plan(plan)

        if route in {"structured", "hybrid"}:
            data_result = run_data_agent(plan)

        if route in {"rag", "temporal_rag", "hybrid"}:
            research_result = run_research_agent(plan)

        synthesis = run_synthesis_agent(
            user_question=question,
            route=route,
            data_agent_result=data_result,
            research_agent_result=research_result,
        )

        post = apply_post_execution_guardrails(
            route=route,
            answer=synthesis["answer"],
            allowed_evidence_ids=_extract_allowed_evidence_ids(
                research_result
            ),
        )

        return {
            "status": synthesis.get("status", "success"),
            "question": question,
            "route": route,
            "plan": plan,
            "answer": synthesis["answer"],
            "citation_validation": synthesis.get("citation_validation"),
            "data_agent_result": data_result,
            "research_agent_result": research_result,
            "guardrails": {
                "input": input_check,
                "plan": plan_check,
                "post": post,
            },
            "total_latency_ms": _elapsed_ms(),
        }

    except Exception as exc:
        # Return a well-formed response instead of an HTTP 500 so callers
        # can see which stage failed. The traceback goes to the serving logs.
        traceback.print_exc()

        return {
            "status": "error",
            "question": question,
            "route": route,
            "plan": plan,
            "answer": f"The request could not be completed: {type(exc).__name__}: {exc}",
            "citation_validation": None,
            "data_agent_result": data_result,
            "research_agent_result": research_result,
            "guardrails": None,
            "total_latency_ms": _elapsed_ms(),
        }
