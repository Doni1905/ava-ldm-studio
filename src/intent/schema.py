"""
src/intent/schema.py
====================
Schema definitions and validation for AVA Intent and Entity Extraction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class IntentExtractionResult:
    """The structured output format for the intent and entity extraction."""
    intent: str
    entities: Dict[str, str] = field(default_factory=dict)
    confidence: float = 0.0
    ambiguous: bool = False
    possible_intents: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "confidence": self.confidence,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "IntentExtractionResult":
        # Basic JSON schema validation
        if "intent" not in data:
            raise ValueError("Missing 'intent' field")
        if "entities" not in data:
            raise ValueError("Missing 'entities' field")
        if not isinstance(data["entities"], dict):
            raise ValueError("'entities' must be a dictionary")
        if "confidence" not in data:
            raise ValueError("Missing 'confidence' field")
        if not isinstance(data["confidence"], (int, float)):
            raise ValueError("'confidence' must be a number")
            
        return cls(
            intent=data["intent"],
            entities=data["entities"],
            confidence=float(data["confidence"])
        )
