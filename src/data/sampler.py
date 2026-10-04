"""
src/data/sampler.py
====================
Stratified sample selection for the AVA LDM working dataset.

Goal
----
Select exactly ``target_n`` (default 5,000) rows from the filtered
DataFrame WITHOUT simply taking the first N rows.

Algorithm
---------
The selection uses **multi-dimensional stratified proportional sampling**:

1.  Build a stratification key combining:
        gender × age_group × area × scenario × task_name
    (district and speaker_id are used for secondary diversity, not as
    primary strata, because they have very high cardinality.)

2.  Compute the natural frequency of each stratum cell.

3.  Allocate ``target_n`` seats proportionally to each cell
    (``floor`` allocation with remainder distributed to the largest cells).

4.  Within each cell, select the allocated number of rows by
    **maximising speaker diversity** — i.e. prefer rows from speakers
    not yet sampled in that cell, using a greedy round-robin approach.

5.  If any cell is under-populated (fewer rows than its allocation),
    the remaining seats are distributed to other cells proportionally.

6.  District diversity check: after sampling, compute the Gini
    coefficient across districts and log a warning if it exceeds 0.7
    (indicating a very skewed geographic distribution).

This ensures:
  - All 7 demographic/linguistic dimensions are represented.
  - Speaker diversity is maximised within each stratum.
  - The selection is deterministic (seeded RNG) and reproducible.
  - The first-N bias is completely eliminated.

Reproducibility
---------------
All random operations use ``numpy.random.default_rng(seed)`` where
``seed`` is read from configs/dataset.yaml → selection.random_seed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class StratifiedSampler:
    """
    Multi-dimensional stratified proportional sampler.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration from configs/dataset.yaml.
    """

    # Primary strata — used to form cells
    _PRIMARY_STRATA = ["gender", "age_group", "area", "scenario", "task_name"]
    # Secondary diversity axes — used for within-cell speaker selection
    _SECONDARY = ["speaker_id", "district"]

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        sel = cfg["selection"]
        self.target_n: int = sel["target_n"]
        self.seed: int = sel["random_seed"]
        self.rng = np.random.default_rng(self.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Select ``target_n`` rows from *df* using stratified proportional
        sampling. See module docstring for the full algorithm.

        Parameters
        ----------
        df : pd.DataFrame
            Filtered metadata DataFrame (output of MetadataSelector).

        Returns
        -------
        pd.DataFrame
            Selected rows with a new ``stratum_key`` column added for
            traceability and a ``selection_rank`` column (within-stratum
            rank used for reproducibility).
        """
        if df.empty:
            raise ValueError("Cannot sample from an empty DataFrame.")

        n = min(self.target_n, len(df))
        if n < self.target_n:
            logger.warning(
                "DataFrame has only %d rows — target_n=%d cannot be reached.",
                len(df), self.target_n,
            )

        logger.info(
            "StratifiedSampler — selecting %d from %d rows (seed=%d)",
            n, len(df), self.seed,
        )

        # Identify available strata columns
        available_strata = [c for c in self._PRIMARY_STRATA if c in df.columns]
        missing_strata = [c for c in self._PRIMARY_STRATA if c not in df.columns]
        if missing_strata:
            logger.warning("Strata columns missing from DataFrame: %s", missing_strata)

        # Build stratum key
        df = df.copy()
        df["stratum_key"] = self._build_stratum_key(df, available_strata)

        # Compute proportional allocation per cell
        allocation = self._proportional_allocation(df, n)
        logger.debug("Stratum allocations: %d cells", len(allocation))

        # Select rows cell by cell
        selected_parts: list[pd.DataFrame] = []
        for stratum_key, quota in allocation.items():
            cell_df = df[df["stratum_key"] == stratum_key]
            picked = self._select_within_cell(cell_df, quota)
            selected_parts.append(picked)

        result = pd.concat(selected_parts, ignore_index=True)

        # Shuffle final selection (avoid speaker clustering from concat order)
        result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        result["selection_rank"] = range(1, len(result) + 1)

        # Trim to exactly target_n (may be slightly over due to rounding)
        result = result.iloc[:n].reset_index(drop=True)

        self._log_sample_summary(result, available_strata)
        self._check_district_gini(result)

        return result

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_stratum_key(self, df: pd.DataFrame, cols: list[str]) -> pd.Series:
        """Concatenate stratum column values into a single key string."""
        if not cols:
            return pd.Series(["__all__"] * len(df), index=df.index)
        parts = []
        for col in cols:
            parts.append(df[col].astype(str).str.strip().str.lower())
        return parts[0].str.cat(parts[1:], sep="|")

    def _proportional_allocation(
        self, df: pd.DataFrame, target: int
    ) -> dict[str, int]:
        """
        Compute how many rows to take from each stratum cell, proportional
        to the cell's natural frequency, with remainder going to largest cells.
        """
        counts = df["stratum_key"].value_counts()
        total = counts.sum()
        # Proportional (floor)
        fracs = (counts / total) * target
        floors = fracs.apply(int)
        remainder = target - floors.sum()

        # Distribute remainder to cells with highest fractional parts
        fractional = fracs - floors
        top_cells = fractional.nlargest(remainder).index
        for cell in top_cells:
            floors[cell] += 1

        # Cap each cell at its actual count (can't sample more than available)
        allocation: dict[str, int] = {}
        deficit = 0
        for cell, quota in floors.items():
            available = int(counts[cell])
            if quota > available:
                deficit += quota - available
                allocation[cell] = available
            else:
                allocation[cell] = int(quota)

        # Redistribute deficit to cells that still have spare capacity
        if deficit > 0:
            spare_cells = {
                c: int(counts[c]) - allocation[c]
                for c in counts.index
                if int(counts[c]) - allocation[c] > 0
            }
            for cell, spare in sorted(spare_cells.items(), key=lambda x: -x[1]):
                if deficit == 0:
                    break
                extra = min(deficit, spare)
                allocation[cell] += extra
                deficit -= extra

        logger.info(
            "Proportional allocation: %d cells, total allocated=%d",
            len(allocation), sum(allocation.values()),
        )
        return allocation

    def _select_within_cell(self, cell_df: pd.DataFrame, quota: int) -> pd.DataFrame:
        """
        Select *quota* rows from *cell_df* maximising speaker diversity.

        Strategy: greedy round-robin across speakers (sorted by speaker_id
        to make it deterministic), then fill remaining slots randomly if needed.
        """
        if len(cell_df) <= quota:
            return cell_df.copy()

        if "speaker_id" not in cell_df.columns:
            # Fallback: random sample
            return cell_df.sample(n=quota, random_state=self.seed)

        # Group rows by speaker
        speakers = cell_df.groupby("speaker_id", sort=True)
        # One bucket per speaker (shuffled within bucket for variety)
        buckets: list[list[int]] = []
        for _, grp in speakers:
            idxs = grp.index.tolist()
            # Shuffle within speaker to avoid biasing toward specific rows
            self.rng.shuffle(idxs)
            buckets.append(idxs)

        # Round-robin pick — cycle through speakers taking 1 row each turn
        selected_idxs: list[int] = []
        bucket_pos = [0] * len(buckets)
        round_idx = 0
        while len(selected_idxs) < quota:
            any_progress = False
            for b_i, bucket in enumerate(buckets):
                if len(selected_idxs) >= quota:
                    break
                pos = bucket_pos[b_i]
                if pos < len(bucket):
                    selected_idxs.append(bucket[pos])
                    bucket_pos[b_i] += 1
                    any_progress = True
            if not any_progress:
                break
            round_idx += 1

        return cell_df.loc[selected_idxs].copy()

    def _log_sample_summary(self, df: pd.DataFrame, strata_cols: list[str]) -> None:
        logger.info("=== Sample Summary ===")
        logger.info("  Total selected: %d", len(df))
        if "speaker_id" in df.columns:
            logger.info("  Unique speakers: %d", df["speaker_id"].nunique())
        for col in strata_cols:
            if col in df.columns:
                dist = df[col].value_counts().to_dict()
                logger.info("  %s: %s", col, dist)

    def _check_district_gini(self, df: pd.DataFrame) -> None:
        """Log a warning if geographic distribution is very skewed."""
        if "district" not in df.columns:
            return
        counts = df["district"].value_counts().values
        if len(counts) <= 1:
            return
        gini = self._gini(counts)
        logger.info("  District Gini coefficient: %.3f", gini)
        if gini > 0.7:
            logger.warning(
                "District distribution is highly skewed (Gini=%.3f). "
                "Consider increasing target_n or broadening filters.",
                gini,
            )

    @staticmethod
    def _gini(values: np.ndarray) -> float:
        """Compute the Gini coefficient (0 = perfectly equal)."""
        if values.sum() == 0:
            return 0.0
        v = np.sort(values)
        n = len(v)
        cumsum = np.cumsum(v)
        return float((2 * cumsum.sum() / (n * v.sum())) - (n + 1) / n)
