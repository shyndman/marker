from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from surya.common.progress import ProgressEvent as SuryaProgressEvent


@dataclass(frozen=True)
class LLMProgressEvent:
    operation: Literal["llm"]
    completed: int
    total: int


ProgressHandler = Callable[[SuryaProgressEvent | LLMProgressEvent], None]
