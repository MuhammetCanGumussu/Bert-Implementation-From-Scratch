
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class ModelInput:
    x: np.ndarray
    y: np.ndarray
    attention_mask: np.ndarray
    segment_ids: np.ndarray

@dataclass
class FillInput:
    mask_word_array: np.ndarray 
    replace_word_array: np.ndarray 
    identity_word_array: np.ndarray 


@dataclass
class VisualizeInputAB:
    ab: pd.Series