"""
src/intent/entity_extractor.py
==============================
Extracts entities required for specific intents without hallucination.
Leverages the existing SemanticPreserver to securely identify entities.
"""

from __future__ import annotations

import logging
from typing import Dict, List

from src.ldm.semantic_preserver import SemanticPreserver, ExtractionResult

logger = logging.getLogger(__name__)

class EntityExtractor:
    """Extracts required and optional entities for a given intent."""
    
    def __init__(self, preserver: SemanticPreserver = None):
        self._preserver = preserver or SemanticPreserver.default()

    def extract(self, text: str, original_text: str = None) -> Dict[str, str]:
        """
        Extracts entities present in the text.
        We run extraction on the normalized text. 
        For best results with Tamil names, the original_text can be passed if needed,
        but typically normalized text contains the preserved entities.
        """
        result: ExtractionResult = self._preserver.extract(text)
        
        entities = {}
        
        # Map SemanticPreserver entity types to intent entity fields
        
        # Time / Date
        if result.time_refs:
            # Simple heuristic: first time ref is date/time
            entities["date"] = result.time_refs[0]
            if len(result.time_refs) > 1:
                entities["time"] = result.time_refs[1]
                
        # Persons
        persons = [a.value for a in result.anchors if a.entity_type == "person"]
        if persons:
            entities["person"] = persons[0]
            
        # Apps
        apps = [a.value for a in result.anchors if a.entity_type == "app"]
        if apps:
            entities["app"] = apps[0]
            
        # Device settings
        devices = [a.value for a in result.anchors if a.entity_type == "device"]
        if devices:
            entities["device_setting"] = devices[0]
            
        # Tasks
        tasks = [a.value for a in result.anchors if a.entity_type == "task"]
        if tasks:
            entities["task"] = tasks[0]
            
        return entities

    @classmethod
    def default(cls) -> "EntityExtractor":
        return cls()
