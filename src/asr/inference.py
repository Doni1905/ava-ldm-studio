"""
src/asr/inference.py
=====================
Orchestrator for running ASR inference over a dataset.
Handles batching, incremental saving, and error recovery.
"""

import os
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List
import soundfile as sf
import json

from .asr_engine import ASREngine
from .postprocess import normalize_text
# Try to reuse the AudioLoader, but if not imported, fallback to soundfile directly for simplicity.
try:
    from src.audio.loader import AudioLoader, AudioLoadError
    HAS_AUDIO_LOADER = True
except ImportError:
    HAS_AUDIO_LOADER = False

logger = logging.getLogger(__name__)

class ASRInferencePipeline:
    def __init__(self, engine: ASREngine, config: Dict[str, Any]):
        self.engine = engine
        
        inf_cfg = config.get("inference", {})
        self.batch_size = inf_cfg.get("batch_size", 4)
        self.save_every = inf_cfg.get("save_every", 10)
        self.skip_existing = inf_cfg.get("skip_existing", True)
        
        paths_cfg = config.get("paths", {})
        self.output_csv = Path(paths_cfg.get("predictions_csv", "results/asr/predictions.csv"))
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        
        if HAS_AUDIO_LOADER:
            self.loader = AudioLoader()
            
    def _load_audio(self, file_path: str):
        """Load audio to float32 numpy array, returning (array, sample_rate)."""
        if HAS_AUDIO_LOADER:
            try:
                audio_data = self.loader.load(file_path)
                return audio_data.waveform_mono, audio_data.sample_rate
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                return None, None
        else:
            try:
                data, sr = sf.read(file_path, dtype="float32", always_2d=True)
                mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
                return mono, sr
            except Exception as e:
                logger.error(f"Failed to load {file_path}: {e}")
                return None, None

    def run_inference(self, manifest_path: str, audio_col: str = "audio_hf_path") -> pd.DataFrame:
        """
        Run inference on a manifest file containing audio paths.
        The manifest should have columns like sample_id, audio_hf_path, speaker_id, etc.
        """
        logger.info(f"Loading manifest from {manifest_path}")
        df = pd.read_csv(manifest_path)
        
        if audio_col not in df.columns:
            # Fallback if preprocessing created a different column or we use 'file'
            if "file" in df.columns:
                audio_col = "file"
            else:
                raise ValueError(f"Column '{audio_col}' not found in manifest.")
                
        # Handle existing predictions (Resume)
        existing_results = {}
        if self.skip_existing and self.output_csv.exists():
            try:
                existing_df = pd.read_csv(self.output_csv)
                if "sample_id" in existing_df.columns:
                    for _, row in existing_df.iterrows():
                        existing_results[str(row["sample_id"])] = row.to_dict()
                logger.info(f"Loaded {len(existing_results)} existing predictions. Resuming...")
            except Exception as e:
                logger.warning(f"Could not load existing predictions: {e}")
                
        results = []
        batch_records = []
        batch_audio = []
        batch_srs = []
        
        # Ensure sample_id exists
        if "sample_id" not in df.columns:
            df["sample_id"] = df.index
            
        total_samples = len(df)
        processed_count = 0
        
        for idx, row in df.iterrows():
            sample_id = str(row["sample_id"])
            
            # Skip if already processed
            if self.skip_existing and sample_id in existing_results:
                results.append(existing_results[sample_id])
                processed_count += 1
                continue
                
            audio_path = row[audio_col]
            if not Path(audio_path).exists():
                logger.warning(f"Audio file not found: {audio_path}")
                continue
                
            arr, sr = self._load_audio(audio_path)
            if arr is None:
                continue
                
            batch_records.append(row)
            batch_audio.append(arr)
            batch_srs.append(sr)
            
            # Process batch if full
            if len(batch_records) >= self.batch_size:
                batch_results = self._process_batch(batch_records, batch_audio, batch_srs)
                results.extend(batch_results)
                processed_count += len(batch_results)
                
                # Incremental save
                if (processed_count % self.save_every) < self.batch_size:
                    self._save_results(results)
                    logger.info(f"Processed {processed_count}/{total_samples} samples. Saved checkpoints.")
                    
                batch_records, batch_audio, batch_srs = [], [], []

        # Process remaining
        if batch_records:
            batch_results = self._process_batch(batch_records, batch_audio, batch_srs)
            results.extend(batch_results)
            processed_count += len(batch_results)
            
        # Final save
        self._save_results(results)
        logger.info(f"Inference complete. Processed {processed_count}/{total_samples} samples.")
        
        return pd.DataFrame(results)
        
    def _process_batch(self, records, audio_arrays, sample_rates):
        t0 = time.time()
        
        try:
            transcriptions = self.engine.transcribe_batch(audio_arrays, sample_rates)
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            # Append empty results with error
            transcriptions = [{"text": "", "error": str(e)} for _ in records]
            
        latency = (time.time() - t0) / max(len(records), 1)
        
        batch_results = []
        for rec, trans in zip(records, transcriptions):
            pred_text = normalize_text(trans.get("text", ""))
            
            res = {
                "sample_id": rec.get("sample_id", ""),
                "reference": rec.get("verbatim", ""), # Adjust based on dataset schema
                "prediction": pred_text,
                "speaker_id": rec.get("speaker_id", ""),
                "district": rec.get("district", ""),
                "duration": rec.get("duration", 0.0),
                "inference_time": latency,
                "timestamps": json.dumps(trans.get("chunks", [])) if trans.get("chunks") else ""
            }
            batch_results.append(res)
            
        return batch_results

    def _save_results(self, results: List[Dict]):
        df = pd.DataFrame(results)
        # Reorder columns as requested
        cols = ["sample_id", "reference", "prediction", "speaker_id", "district", "duration", "inference_time", "timestamps"]
        # Only keep columns that exist
        cols = [c for c in cols if c in df.columns]
        df.to_csv(self.output_csv, index=False, columns=cols)

