"""Transcript clean-up: strip the wake phrase, drop Whisper's silence hallucinations."""

from __future__ import annotations

import re
from difflib import SequenceMatcher

_TOKEN = re.compile(r"\S+")


def _norm(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower())


def _similar(a: str, b: str, cutoff: float) -> bool:
    if a == b:
        return True
    return SequenceMatcher(None, a, b).ratio() >= cutoff


def strip_wake_phrase(text: str, phrases: list[str], cutoff: float = 0.75) -> str:
    """Remove leading occurrences of any wake phrase, fuzzily.

    "Hey Jarvis, turn off the lights" -> "Turn off the lights"
    "Hey, Jervis. What time is it?"   -> "What time is it?"
    """
    tokens = _TOKEN.findall(text)
    norms = [_norm(t) for t in tokens]
    targets = sorted(([_norm(w) for w in p.split()] for p in phrases if p.strip()),
                     key=len, reverse=True)
    i = 0
    changed = True
    while changed:
        changed = False
        while i < len(norms) and norms[i] == "":
            i += 1
        for target in targets:
            n = len(target)
            window = norms[i : i + n]
            if len(window) == n and all(_similar(a, b, cutoff) for a, b in zip(window, target)):
                i += n
                changed = True
                break
    if i == 0:
        return text.strip()
    rest = " ".join(tokens[i:]).lstrip(",.;:!?-– ").strip()
    return rest[:1].upper() + rest[1:] if rest else ""


def is_hallucination(text: str, drop_phrases: list[str]) -> bool:
    """True if the transcript is empty or only one of Whisper's stock phrases."""
    norm = " ".join(_norm(t) for t in _TOKEN.findall(text) if _norm(t))
    if not norm:
        return True
    return any(norm == " ".join(_norm(w) for w in p.split()) for p in drop_phrases)
