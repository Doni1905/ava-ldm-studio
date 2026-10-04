"""
src/asr/asr_engine.py
======================
Abstract base class for ASR Engines in AVA LDM Studio.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import numpy as np

class ASREngine(ABC):
    """
    Abstract interface for Speech-to-Text models.
    """
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ASR engine with the given configuration.
        """
        pass

    @abstractmethod
    def transcribe_batch(
        self, 
        audio_arrays: List[np.ndarray], 
        sample_rates: List[int]
    ) -> List[Dict[str, Any]]:
        """
        Transcribe a batch of audio waveforms.
        
        Parameters
        ----------
        audio_arrays : List[np.ndarray]
            List of 1D float32 numpy arrays containing mono audio.
        sample_rates : List[int]
            List of sample rates for each audio array.
            
        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries containing:
            - "text": str (the transcribed text)
            - "chunks": Optional[List[Dict]] (timestamps, if requested and supported)
        """
        pass

    @abstractmethod
    def get_target_sample_rate(self) -> int:
        """
        Return the expected sample rate for this ASR model (e.g., 16000).
        """
        pass

