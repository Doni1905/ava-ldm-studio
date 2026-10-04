"""
src/dialect/labels.py
======================
Dialect label registry and district→dialect mapping for AVA LDM Studio.

Design note
-----------
This module is the single source of truth for label definitions.  It reads
from ``configs/dialect.yaml`` so that adding or renaming dialect labels never
requires changes to Python source files.

District ≠ Dialect — explicit documentation
--------------------------------------------
The IndicVoices-R Tamil dataset annotates each utterance with:

  - ``district``:   Tamil Nadu administrative district (e.g. "Coimbatore").
  - ``area``:       "Urban" | "Rural".

Neither of these is a dialect label.  Tamil dialects are sociolinguistic
varieties defined by phonology, morphology, and vocabulary — they span
administrative boundaries.  This module provides an explicit, configurable
mapping from geographic districts to dialect regions, based on:

  1. Annamalai, E. (2011). *Nativizing English in India.* — Regional dialect
     zones of Tamil.
  2. Schiffman, H. F. (1998). *Linguistic Culture and Language Policy.*
     — Tamil dialect continuum.
  3. The prototype dialect markers in ``src/lib/ldm/processor.ts`` which
     already defined four operational dialect regions with lexical evidence:
     Chennai (Madras Bashai), Madurai, Kongu/Coimbatore, Nellai.

This mapping is a best-effort approximation.  When dialect-annotated
training data becomes available, update ``configs/dialect.yaml`` —
not this file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# ---------------------------------------------------------------------------
# Label Registry
# ---------------------------------------------------------------------------

class DialectLabelRegistry:
    """
    Loads and validates dialect label configuration from YAML.

    Attributes
    ----------
    names : list[str]
        Ordered list of dialect label strings, e.g. ["Chennai", "Madurai", ...]
    label2idx : dict[str, int]
        Map from label string to integer index.
    idx2label : dict[int, str]
        Map from integer index to label string.
    default_label : str
        Label used for districts not present in the mapping.
    district_map : dict[str, str]
        Mapping from district name → dialect label.
    num_classes : int
        Total number of dialect classes.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        lbl_cfg = config.get("labels", config)

        self.names: List[str] = lbl_cfg["names"]
        self.default_label: str = lbl_cfg.get("default_label", self.names[-1])
        self.district_map: Dict[str, str] = lbl_cfg.get("district_to_dialect", {})

        # Validate: default_label must be in names
        if self.default_label not in self.names:
            raise ValueError(
                f"default_label '{self.default_label}' not in labels.names: {self.names}"
            )

        # Validate: every district mapping target must be in names
        for district, dialect in self.district_map.items():
            if dialect not in self.names:
                raise ValueError(
                    f"district_to_dialect['{district}'] = '{dialect}' not in names: {self.names}"
                )

        self.label2idx: Dict[str, int] = {lbl: i for i, lbl in enumerate(self.names)}
        self.idx2label: Dict[int, str] = {i: lbl for i, lbl in enumerate(self.names)}

    @property
    def num_classes(self) -> int:
        return len(self.names)

    def district_to_dialect(self, district: Optional[str]) -> str:
        """
        Map a geographic district name to a dialect label.

        If the district is not found in the mapping, returns ``default_label``.

        Parameters
        ----------
        district : str | None
            District name as it appears in the dataset (case-insensitive lookup).
        """
        if not district:
            return self.default_label
        # Try exact match first, then title-case
        for key in [district, district.strip().title()]:
            if key in self.district_map:
                return self.district_map[key]
        return self.default_label

    def label_to_idx(self, label: str) -> int:
        if label not in self.label2idx:
            raise KeyError(f"Unknown dialect label: '{label}'. Known: {self.names}")
        return self.label2idx[label]

    def idx_to_label(self, idx: int) -> str:
        if idx not in self.idx2label:
            raise KeyError(f"Unknown dialect index: {idx}. Valid: 0..{self.num_classes - 1}")
        return self.idx2label[idx]

    def map_manifest_column(self, df, district_col: str = "district") -> "pd.Series":
        """
        Apply district→dialect mapping to a manifest DataFrame column.

        Returns a new Series with dialect labels.
        """
        return df[district_col].apply(self.district_to_dialect)

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "DialectLabelRegistry":
        """Load from a YAML config file."""
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(config=cfg)

    def __repr__(self) -> str:
        return (
            f"DialectLabelRegistry(num_classes={self.num_classes}, "
            f"labels={self.names}, default='{self.default_label}')"
        )
