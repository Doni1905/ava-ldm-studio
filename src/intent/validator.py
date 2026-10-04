"""
src/intent/validator.py
=======================
Coordinates intent classification and entity extraction, producing validated JSON output.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

from .classifier import IntentClassifier, UNKNOWN_INTENT
from .entity_extractor import EntityExtractor
from .schema import IntentExtractionResult

logger = logging.getLogger(__name__)

_DEFAULT_INTENT_CONFIG = Path("configs/intents.yaml")


class IntentValidator:
    """Coordinates intent classification and entity extraction."""
    
    def __init__(self, classifier: IntentClassifier = None, extractor: EntityExtractor = None, config_path: Optional[Path | str] = None):
        self._classifier = classifier or IntentClassifier.default()
        self._extractor = extractor or EntityExtractor.default()
        
        path = Path(config_path) if config_path else self._classifier._root / _DEFAULT_INTENT_CONFIG
        self._intent_schemas = self._load_schemas(path)

    def _load_schemas(self, path: Path) -> Dict[str, Dict[str, Any]]:
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data.get("intents", {})

    def process(self, normalized_text: str) -> IntentExtractionResult:
        """Process normalized text and return structured intent result."""
        
        # 1. Classify Intent
        intent_name, confidence, is_ambiguous, possible = self._classifier.classify(normalized_text)
        
        # 2. Extract Entities
        entities = self._extractor.extract(normalized_text)
        
        # 3. Filter entities based on intent schema
        filtered_entities = {}
        if intent_name in self._intent_schemas:
            schema = self._intent_schemas[intent_name]
            allowed_entities = set(schema.get("required_entities", []) + schema.get("optional_entities", []))
            
            for k, v in entities.items():
                if k in allowed_entities:
                    filtered_entities[k] = v
        else:
            # If unknown intent, we still return extracted entities for debugging or general queries
            filtered_entities = entities

        return IntentExtractionResult(
            intent=intent_name,
            entities=filtered_entities,
            confidence=confidence,
            ambiguous=is_ambiguous,
            possible_intents=possible
        )

    @classmethod
    def default(cls) -> "IntentValidator":
        return cls()
