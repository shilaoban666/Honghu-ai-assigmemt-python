"""Built-in tools for the skill system — mirrors the Java builtin tools.

Provides: current_time, calculator, rag_search, and other utility tools
that are always available to the model.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from langchain_core.tools import tool


@tool
def current_time() -> str:
    """Get the current date and time in ISO 8601 format. Use when the user asks about the current time, today's date, or needs a timestamp."""
    return datetime.now(timezone.utc).isoformat()


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression. Supports +, -, *, /, ** (power), sqrt, sin, cos, tan, log, abs, pi, e.
    Example input: "2 + 3 * 4" or "sqrt(16) + 2**3"
    """
    # Safe eval with math functions
    allowed_names = {
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "log": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "abs": abs,
        "pi": math.pi,
        "e": math.e,
        "ceil": math.ceil,
        "floor": math.floor,
        "round": round,
        "pow": pow,
    }

    # Sanitize input
    expression = expression.strip()
    if not expression:
        return "Error: empty expression"

    # Only allow safe characters
    if not re.match(r"^[\d\s+\-*/().,%\^a-z_]+$", expression):
        return "Error: expression contains invalid characters"

    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def word_count(text: str) -> str:
    """Count words, characters, and lines in the given text. Returns a summary with word count, character count, and line count."""
    lines = text.count("\n") + 1 if text else 0
    words = len(text.split()) if text else 0
    chars = len(text)
    return f"Words: {words}, Characters: {chars}, Lines: {lines}"


@tool
def text_case_transform(text: str, operation: str) -> str:
    """Transform text case. Operation can be: upper, lower, title, capitalize, or sentence.
    Example: text_case_transform("hello world", "upper") → "HELLO WORLD"
    """
    if operation == "upper":
        return text.upper()
    elif operation == "lower":
        return text.lower()
    elif operation == "title":
        return text.title()
    elif operation == "capitalize":
        return text.capitalize()
    elif operation == "sentence":
        return ". ".join(s.capitalize() for s in text.split(". "))
    else:
        return f"Unknown operation: {operation}. Use: upper, lower, title, capitalize, or sentence."


# All built-in tools list
BUILTIN_TOOLS = [
    current_time,
    calculator,
    word_count,
    text_case_transform,
]
