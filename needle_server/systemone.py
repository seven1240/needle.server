"""POST /v1/systemone: evaluate a state against typed questions.

This mirrors the TypeSafe AI "System One" (Jev) evaluation API. Each question is
one of three types -- noul, choice or score -- and the endpoint responds with one
structured answer per question.

needle.server runs a tool-calling model: it returns function_calls plus a single
calibrated confidence in [0, 1], with no per-token logits. So each System One
question is translated into a synthetic tool call, the model's binary answer is
read back, and the confidence is folded in to approximate probabilities.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from . import config


class RequestError(Exception):
    """A validation failure in a request; the route maps it to a 400 response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Request schema
# --------------------------------------------------------------------------- #

class NoulQuestion(BaseModel):
    type: Literal["noul"]
    instructions: str | dict | list
    criteria: dict | None = None


class ChoiceQuestion(BaseModel):
    type: Literal["choice"]
    instructions: str | dict | list
    criteria: dict
    # criteria is required and maps option key -> description; 1..255 options.


class ScoreQuestion(BaseModel):
    type: Literal["score"]
    instructions: str | dict | list
    criteria: list
    # criteria is required and is an ordered list of 2..10 level descriptions.


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    state: str | dict | list
    model: str = "jev-latest"  # accepted for compatibility; the server always runs needle3
    questions: dict[str, Question]


def _to_text(field: Any) -> str:
    if isinstance(field, str):
        return field
    return json.dumps(field, separators=(",", ":"), ensure_ascii=False)


def _criteria_text(criteria: Any) -> str:
    if isinstance(criteria, str):
        return criteria
    if isinstance(criteria, dict):
        return "; ".join(f"{k}: {_to_text(v) if v is not None else ''}" for k, v in criteria.items())
    return "; ".join(_to_text(item) for item in criteria)


# --------------------------------------------------------------------------- #
# Translation: each question becomes one synthetic tool + a system descriptor.
# --------------------------------------------------------------------------- #

def build_tool(index: int, qid: str, question: Question) -> dict[str, Any]:
    """One tool per question, named by index so arbitrary caller ids map back
    reliably. The rubric (criteria) is folded into the description and schema."""
    safe = f"systemone_{index}"
    if isinstance(question, NoulQuestion):
        description = (f"Decide whether the state satisfies a yes/no condition for question "
                       f"{qid!r}: {_to_text(question.instructions)}.")
        if question.criteria:
            description += f" Yes means: {_to_text(question.criteria.get('true'))}. No means: {_to_text(question.criteria.get('false'))}."
        return {
            "name": safe,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"answer": {"type": "boolean"}},
                "required": ["answer"],
            },
        }

    if isinstance(question, ChoiceQuestion):
        options = list(question.criteria.keys())
        description = (f"Pick the single best-matching option for question {qid!r}: "
                       f"{_to_text(question.instructions)}. Options: {_criteria_text(question.criteria)}.")
        return {
            "name": safe,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {"choice": {"type": "string", "enum": options}},
                "required": ["choice"],
            },
        }

    # ScoreQuestion
    levels = list(range(len(question.criteria)))
    description = (f"Rate the state on an ordered scale for question {qid!r}: "
                   f"{_to_text(question.instructions)}. Levels: {_criteria_text(question.criteria)}.")
    return {
        "name": safe,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {"level": {"type": "integer", "minimum": 0, "maximum": len(question.criteria) - 1}},
            "required": ["level"],
        },
    }


def build_system(pairs: list[tuple[str, Question]]) -> str | None:
    if not pairs:
        return None
    lines = ["Evaluate the given state and answer every question with exactly one tool call. "
             "The questions are:"]
    for index, (qid, q) in enumerate(pairs):
        lines.append(f"{index}: {qid} ({q.type}) {_to_text(q.instructions)}")
    return "\n".join(lines)


def serialize_state(state: str | dict | list) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Parsing: read function_calls back into System One answers.
# --------------------------------------------------------------------------- #

def _other_prob(c: float, count: int) -> float:
    """Confidence is a single calibrated score; this approximates the share each
    non-chosen item receives when one item holds `c`. Only meaningful for
    count >= 2; callers guard that before use."""
    return max(0.0, (1.0 - c) / (count - 1))


def parse_answers(function_calls: list[dict], pairs: Iterable[tuple[str, Question]], confidence: float) -> dict:
    """Map each question id -> its answer. Answers not covered by a call are
    reported with all probability on the model's default state and low confidence."""
    calls = {call.get("name"): call.get("arguments", {}) for call in function_calls if isinstance(call.get("name"), str)}
    answers: dict[str, Any] = {}
    for index, (qid, q) in enumerate(pairs):
        safe = f"systemone_{index}"
        args = calls.get(safe, {})
        answers[qid] = build_answer(q, args, confidence)
    return answers


def build_answer(question: Question, args: dict[str, Any], confidence: float) -> dict[str, Any]:
    if isinstance(question, NoulQuestion):
        yes = bool(args.get("answer"))
        noul = confidence if yes else 1.0 - confidence
        return {"type": "noul", "noul": round(noul, 4)}

    if isinstance(question, ChoiceQuestion):
        options = list(question.criteria.keys())
        chosen = args.get("choice")
        if chosen is None or chosen not in options:
            chosen = options[0] if options else None
        count = max(2, len(options))  # options never empty: validate() enforces >= 1
        probs = {opt: round(_other_prob(confidence, count), 4) for opt in options}
        if chosen is not None:
            probs[chosen] = round(confidence, 4)
        return {"type": "choice", "choice": chosen, "probabilities": probs, "confidence": round(confidence, 4)}

    # ScoreQuestion
    levels = len(question.criteria)
    idx = args.get("level")
    if not isinstance(idx, int) or not (0 <= idx < levels):
        idx = 0
    probs = {str(j): round(_other_prob(confidence, levels), 4) for j in range(levels)}
    probs[str(idx)] = round(confidence, 4)
    score = sum(j * probs[str(j)] for j in range(levels))
    legend = {str(j): _to_text(question.criteria[j]) for j in range(levels)}
    return {
        "type": "score",
        "score": round(score, 4),
        "legend": legend,
        "probabilities": probs,
        "confidence": round(confidence, 4),
    }


# --------------------------------------------------------------------------- #
# Orchestration: used by app.py.
# --------------------------------------------------------------------------- #

def validate(request: SystemOneRequest) -> list[tuple[str, Question]]:
    if not request.questions:
        raise RequestError("questions_required", "questions must list at least one question")
    if len(request.questions) > config.MAX_SYSTEMONE_QUESTIONS:
        raise RequestError(
            "too_many_questions",
            f"{len(request.questions)} questions; the limit is {config.MAX_SYSTEMONE_QUESTIONS}")
    state_chars = len(serialize_state(request.state))
    if state_chars > config.MAX_SYSTEMONE_STATE_CHARS:
        raise RequestError(
            "state_too_large",
            f"state is {state_chars} characters; the limit is {config.MAX_SYSTEMONE_STATE_CHARS}")
    pairs: list[tuple[str, Question]] = []
    for qid, q in request.questions.items():
        if isinstance(q, ChoiceQuestion) and not (1 <= len(q.criteria) <= config.MAX_SYSTEMONE_OPTIONS):
            raise RequestError("invalid_options", f"question {qid!r} needs 1..{config.MAX_SYSTEMONE_OPTIONS} "
                               f"options; it has {len(q.criteria)}")
        if isinstance(q, ScoreQuestion) and not (2 <= len(q.criteria) <= config.MAX_SYSTEMONE_LEVELS):
            raise RequestError("invalid_score_levels", f"question {qid!r} needs 2.."
                               f"{config.MAX_SYSTEMONE_LEVELS} levels; it has {len(q.criteria)}")
        if not qid.strip():
            raise RequestError("empty_question_id", "question ids must not be empty")
        pairs.append((qid, q))
    return pairs


def translate(request: SystemOneRequest) -> tuple[str, str | None, str]:
    """Build the needle toolset, system and input for a SystemOne request."""
    pairs = validate(request)
    tools = [build_tool(i, qid, q) for i, (qid, q) in enumerate(pairs)]
    tools_json = json.dumps(tools, separators=(",", ":"), ensure_ascii=False)
    system = build_system(pairs)
    return tools_json, system, serialize_state(request.state)
