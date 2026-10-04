"""
src/intent/classifier.py
========================
Modular rule-based intent classifier.
Loads regex patterns from configs/intents.yaml and matches normalized text to an intent.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)

_DEFAULT_INTENT_CONFIG = Path("configs/intents.yaml")
UNKNOWN_INTENT = "UNKNOWN_INTENT"


class IntentClassifier:
    """Classifies normalized text into a primary intent with a confidence score."""

    def __init__(self, config_path: Optional[Path | str] = None, project_root: Optional[Path | str] = None):
        self._root = Path(project_root) if project_root else self._find_root()
        path = Path(config_path) if config_path else self._root / _DEFAULT_INTENT_CONFIG

        self._intents = {}
        self._compiled_patterns = {}
        self._load_config(path)

    def _find_root(self) -> Path:
        here = Path(__file__).resolve()
        for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
            if (parent / "configs").exists():
                return parent
        return Path.cwd()

    def _load_config(self, path: Path) -> None:
        if not path.exists():
            logger.warning(f"Intent config not found at {path}")
            return
            
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            
        self._intents = data.get("intents", {})
        
        # Compile patterns
        for intent_name, config in self._intents.items():
            patterns = config.get("patterns", [])
            compiled = []
            for p in patterns:
                try:
                    compiled.append(re.compile(p, re.IGNORECASE))
                except re.error as e:
                    logger.error(f"Invalid regex for intent {intent_name}: {p} - {e}")
            self._compiled_patterns[intent_name] = compiled
            
    def classify(self, text: str) -> Tuple[str, float, bool, List[str]]:
        """
        Classify the intent of the given text.
        
        Returns:
            (intent_name, confidence, is_ambiguous, possible_intents)
        """
        matched_intents = []
        
        for intent_name, patterns in self._compiled_patterns.items():
            for p in patterns:
                if p.search(text):
                    matched_intents.append(intent_name)
                    break # Move to next intent once one pattern matches
                    
        if not matched_intents:
            return UNKNOWN_INTENT, 0.0, False, []
            
        if len(matched_intents) == 1:
            return matched_intents[0], 0.95, False, matched_intents
            
        # Ambiguity resolution: If multiple match, we return the first but flag ambiguity
        # Could implement priorities here, but for now just return the first with lower confidence
        return matched_intents[0], 0.60, True, matched_intents

    @classmethod
    def default(cls, project_root: Optional[str | Path] = None) -> "IntentClassifier":
        return cls(project_root=project_root)
