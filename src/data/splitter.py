"""
src/data/splitter.py
=====================
Speaker-disjoint train / validation / test split creator.

Design principles
-----------------
1.  **Speaker disjointness (hard constraint)**: No speaker_id appears
    in more than one split. This is the most important property for
    evaluating generalisation to unseen speakers.

2.  **Proportional split**: The default 80 / 10 / 10 ratio is achieved
    at the *speaker* level, not the *utterance* level — since different
    speakers may contribute different numbers of utterances.

3.  **Stratification within splits**: After assigning speakers to splits,
    we verify and log the distribution of gender, age_group, area, and
    scenario across splits to ensure each split is reasonably balanced.

4.  **Reproducibility**: All random operations use a seeded RNG.

Algorithm (speaker-level splitting)
-------------------------------------
1.  Compute the set of unique speakers.
2.  For each speaker compute their aggregate demographic profile
    (most common gender, age_group, area, scenario).
3.  Sort speakers by profile to group similar speakers together.
4.  Shuffle speakers WITHIN each demographic stratum using the seeded RNG.
5.  Assign speakers round-robin to splits according to the target fractions.
    This preserves approximate demographic balance across splits.
6.  Map split assignments back to utterance rows.
7.  Verify no speaker appears in more than one split (assert).
8.  Log per-split statistics.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SpeakerDisjointSplitter:
    """
    Creates speaker-disjoint train / validation / test splits.

    Parameters
    ----------
    cfg : dict
        Full pipeline configuration from configs/dataset.yaml.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        spl = cfg["splitting"]
        self.train_frac: float = spl["train_frac"]
        self.val_frac:   float = spl["val_frac"]
        self.test_frac:  float = spl["test_frac"]
        self.strat_by:   list[str] = spl.get("stratify_splits_by", [])
        self.seed: int = cfg["selection"]["random_seed"]
        self.rng = np.random.default_rng(self.seed)

        total = self.train_frac + self.val_frac + self.test_frac
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Split fractions must sum to 1.0, got {total:.6f}"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """
        Assign every row in *df* to train / val / test ensuring no
        speaker appears in more than one split.

        Returns
        -------
        dict with keys 'train', 'val', 'test'.
        """
        if "speaker_id" not in df.columns:
            logger.warning(
                "'speaker_id' column missing — falling back to random split "
                "(speaker disjointness CANNOT be guaranteed)."
            )
            return self._random_split(df)

        logger.info("=== SPEAKER-DISJOINT SPLIT ===")
        logger.info(
            "Target fractions — train: %.0f%%  val: %.0f%%  test: %.0f%%",
            self.train_frac * 100, self.val_frac * 100, self.test_frac * 100,
        )

        # Build speaker → split assignment
        speaker_split = self._assign_speakers_to_splits(df)

        # Map assignments to utterance rows
        df = df.copy()
        df["split"] = df["speaker_id"].map(speaker_split)
        unassigned = df["split"].isna().sum()
        if unassigned:
            logger.warning("%d rows have unassigned speakers — assigning to train.", unassigned)
            df["split"] = df["split"].fillna("train")

        splits = {
            "train": df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True),
            "val":   df[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True),
            "test":  df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True),
        }

        self._log_split_statistics(splits)
        return splits

    def verify_disjoint(self, splits: dict[str, pd.DataFrame]) -> None:
        """
        Assert that no speaker_id appears in more than one split.
        Raises AssertionError if the constraint is violated.
        Logs a SUCCESS message if it holds.
        """
        if not all("speaker_id" in df.columns for df in splits.values()):
            logger.warning("Cannot verify disjointness — 'speaker_id' column missing.")
            return

        train_spk = set(splits["train"]["speaker_id"].unique())
        val_spk   = set(splits["val"]["speaker_id"].unique())
        test_spk  = set(splits["test"]["speaker_id"].unique())

        tv = train_spk & val_spk
        tt = train_spk & test_spk
        vt = val_spk   & test_spk

        violations = tv | tt | vt
        if violations:
            raise AssertionError(
                f"Speaker disjointness VIOLATED! "
                f"Overlapping speakers: {sorted(violations)[:10]} …"
            )

        logger.info(
            "✔ Speaker disjointness verified — "
            "train: %d spk | val: %d spk | test: %d spk",
            len(train_spk), len(val_spk), len(test_spk),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assign_speakers_to_splits(self, df: pd.DataFrame) -> dict[str, str]:
        """
        Return a mapping {speaker_id → split_name} such that:
        - Speakers are distributed ≈ train_frac / val_frac / test_frac.
        - Demographic balance is approximately preserved.
        """
        # Compute per-speaker demographic profile (most frequent value)
        strat_cols = [c for c in self.strat_by if c in df.columns]
        agg: dict[str, str] = {c: (c, lambda s: s.mode().iloc[0] if len(s) else "unknown")
                               for c in strat_cols}

        if strat_cols:
            # Build profile per speaker
            profile_df = (
                df.groupby("speaker_id")[strat_cols]
                .agg(lambda s: s.mode().iloc[0] if len(s) > 0 else "unknown")
                .reset_index()
            )
            # Stratified key for sorting
            profile_df["_skey"] = (
                profile_df[strat_cols]
                .astype(str)
                .apply(lambda r: "|".join(r), axis=1)
            )
            profile_df = profile_df.sort_values("_skey").reset_index(drop=True)
            speakers = profile_df["speaker_id"].tolist()
            strata = profile_df["_skey"].tolist()
        else:
            speakers = sorted(df["speaker_id"].unique().tolist())
            strata = ["__all__"] * len(speakers)

        # Shuffle within each stratum group
        shuffled_speakers = self._stratified_shuffle(speakers, strata)

        # Assign to splits
        n = len(shuffled_speakers)
        n_train = round(n * self.train_frac)
        n_val   = round(n * self.val_frac)
        # Remainder goes to test
        n_test  = n - n_train - n_val

        assignment: dict[str, str] = {}
        for i, spk in enumerate(shuffled_speakers):
            if i < n_train:
                assignment[spk] = "train"
            elif i < n_train + n_val:
                assignment[spk] = "val"
            else:
                assignment[spk] = "test"

        logger.info(
            "Speaker assignments — train: %d | val: %d | test: %d",
            n_train, n_val, n_test,
        )
        return assignment

    def _stratified_shuffle(
        self,
        speakers: list[str],
        strata: list[str],
    ) -> list[str]:
        """
        Shuffle speakers within each stratum group independently,
        then interleave stratum groups proportionally.
        """
        from collections import defaultdict
        groups: dict[str, list[str]] = defaultdict(list)
        for spk, skey in zip(speakers, strata):
            groups[skey].append(spk)

        shuffled_groups: list[list[str]] = []
        for skey in sorted(groups):
            grp = groups[skey]
            self.rng.shuffle(grp)
            shuffled_groups.append(grp)

        # Interleave groups proportionally (round-robin by group size)
        result: list[str] = []
        pointers = [0] * len(shuffled_groups)
        while True:
            progress = False
            for g_i, grp in enumerate(shuffled_groups):
                if pointers[g_i] < len(grp):
                    result.append(grp[pointers[g_i]])
                    pointers[g_i] += 1
                    progress = True
            if not progress:
                break
        return result

    def _random_split(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Fallback: pure random split (no speaker guarantee)."""
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        n = len(df)
        n_train = round(n * self.train_frac)
        n_val   = round(n * self.val_frac)
        return {
            "train": df.iloc[:n_train].reset_index(drop=True),
            "val":   df.iloc[n_train:n_train + n_val].reset_index(drop=True),
            "test":  df.iloc[n_train + n_val:].reset_index(drop=True),
        }

    def _log_split_statistics(self, splits: dict[str, pd.DataFrame]) -> None:
        for name, split_df in splits.items():
            spk_count = split_df["speaker_id"].nunique() if "speaker_id" in split_df else "N/A"
            logger.info(
                "  %s: %d utterances, %s speakers",
                name.upper(), len(split_df), spk_count,
            )
            for col in ("gender", "age_group", "area", "scenario"):
                if col in split_df.columns:
                    dist = split_df[col].value_counts().to_dict()
                    logger.debug("    %s/%s: %s", name, col, dist)
