from __future__ import annotations

from typing import Dict, Optional, Protocol, Sequence, Tuple, Union

import torch


RewardMetadata = Dict[str, object]


class RewardAdapter(Protocol):
    """Protocol that reward adapters must implement."""

    def score_tensor(
        self,
        images: torch.Tensor,
        prompts: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        return_metadata: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, RewardMetadata]]:
        ...
