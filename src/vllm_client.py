"""vLLM client placeholder for future scene explanation."""

from __future__ import annotations

from typing import Any


class VLLMClient:
    """Minimal local vLLM client interface.

    The OpenAI-compatible chat-completions implementation is planned for MVP 5.
    """

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def explain_scene(self, scene_state: dict[str, Any], question: str | None = None) -> str:
        """Return a placeholder explanation for the current scene."""

        _ = scene_state, question
        return "vLLM scene explanation is planned for MVP 5."
