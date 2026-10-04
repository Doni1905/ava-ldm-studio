"""
src/llm/prompt_builder.py
=========================
Prompt construction adapter for the AVA Local LLM.

Adapts LDM semantic handoff into structured, model-ready prompts
without exposing raw dialect or slang speech.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

from .base import LDMHandoff


class PromptBuilder:
    """
    Constructs clean, structured prompts for local LLMs from LDM handoff payloads.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are AVA (Accentric Virtual Assistant), a helpful, concise AI assistant. "
        "The user's query has been normalized from dialect/code-mixed Tamil speech into standard English. "
        "Respond accurately, concisely, and helpfully."
    )

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        format_type: str = "chat",
        include_metadata: bool = True,
    ) -> None:
        self.system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self.format_type = format_type.lower()
        self.include_metadata = include_metadata

    def build_messages(self, input_data: Union[str, LDMHandoff, Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Build standard chat messages list (role: 'system' | 'user' | 'assistant').
        Compatible with Hugging Face tokenizer.apply_chat_template.
        """
        handoff = self._coerce_handoff(input_data)
        user_content = self._format_user_content(handoff)

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def build_text(self, input_data: Union[str, LDMHandoff, Dict[str, Any]]) -> str:
        """
        Build plain text prompt formatted according to format_type ('chat', 'instruct', or 'raw').
        """
        handoff = self._coerce_handoff(input_data)
        user_content = self._format_user_content(handoff)

        if self.format_type == "instruct":
            return (
                f"### System:\n{self.system_prompt}\n\n"
                f"### Instruction:\n{user_content}\n\n"
                f"### Response:\n"
            )
        elif self.format_type == "chat":
            return (
                f"<|system|>\n{self.system_prompt}\n"
                f"<|user|>\n{user_content}\n"
                f"<|assistant|>\n"
            )
        else:  # raw
            return f"{self.system_prompt}\n\nUser: {user_content}\nAssistant: "

    def _format_user_content(self, handoff: LDMHandoff) -> str:
        """
        Format the user content with semantic normalized text and optional entity metadata.
        Crucially: does NOT present raw colloquial dialect to the LLM.
        """
        lines = []

        if self.include_metadata and (handoff.intent or handoff.entities):
            context_parts = []
            if handoff.intent and handoff.intent != "UNKNOWN_INTENT":
                context_parts.append(f"Intent: {handoff.intent}")

            if handoff.entities:
                ent_str = ", ".join(f"{k}='{v}'" for k, v in handoff.entities.items() if v)
                if ent_str:
                    context_parts.append(f"Entities: [{ent_str}]")

            if context_parts:
                lines.append(f"[Understood Context: {'; '.join(context_parts)}]")

        lines.append(handoff.normalized_text.strip())
        return "\n".join(lines).strip()

    @staticmethod
    def _coerce_handoff(data: Union[str, LDMHandoff, Dict[str, Any]]) -> LDMHandoff:
        if isinstance(data, LDMHandoff):
            return data
        if isinstance(data, dict):
            return LDMHandoff.from_dict(data)
        if isinstance(data, str):
            return LDMHandoff(normalized_text=data)
        return LDMHandoff(normalized_text=str(data))
