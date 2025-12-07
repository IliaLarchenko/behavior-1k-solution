"""Custom Normalize transform with per-timestamp support.
"""

import dataclasses
import numpy as np

from openpi.transforms import DataTransformFn, DataDict, apply_tree, pad_to_dim
from openpi.shared import array_typing as at
from b1k.shared.normalize import NormStats


@dataclasses.dataclass(frozen=True)
class NormalizeWithPerTimestamp(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    strict: bool = False
    use_per_timestamp: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            for stats in at.tree_leaves(self.norm_stats):
                if isinstance(stats, NormStats) and (stats.q01 is None or stats.q99 is None):
                    raise ValueError("Quantile normalization requires q01 and q99 in norm_stats")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._normalize_quantile if self.use_quantiles else self._normalize,
            strict=self.strict,
        )

    def _normalize(self, x, stats: NormStats):
        # Check if this is actions and we should use per-timestamp normalization
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            # x has shape [..., action_horizon, action_dim] for actions
            # stats.per_timestamp_mean has shape [action_horizon, action_dim]
            mean = stats.per_timestamp_mean[..., : x.shape[-2], : x.shape[-1]]
            std = stats.per_timestamp_std[..., : x.shape[-2], : x.shape[-1]]
            return (x - mean) / (std + 1e-6)
        else:
            # Regular normalization
            mean, std = stats.mean[..., : x.shape[-1]], stats.std[..., : x.shape[-1]]
            return (x - mean) / (std + 1e-6)

    def _normalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        
        # Check if this is actions and we should use per-timestamp normalization
        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            # x has shape [..., action_horizon, action_dim] for actions
            # stats.per_timestamp_q01 has shape [action_horizon, action_dim]
            q01 = stats.per_timestamp_q01[: x.shape[-2], : x.shape[-1]]
            q99 = stats.per_timestamp_q99[: x.shape[-2], : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
        else:
            # Regular quantile normalization
            q01, q99 = stats.q01[..., : x.shape[-1]], stats.q99[..., : x.shape[-1]]
            return (x - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0


@dataclasses.dataclass(frozen=True)
class UnnormalizeWithPerTimestamp(DataTransformFn):
    norm_stats: at.PyTree[NormStats] | None
    use_quantiles: bool = False
    use_per_timestamp: bool = False

    def __post_init__(self):
        if self.norm_stats is not None and self.use_quantiles:
            for stats in at.tree_leaves(self.norm_stats):
                if isinstance(stats, NormStats) and (stats.q01 is None or stats.q99 is None):
                    raise ValueError("Quantile normalization requires q01 and q99 in norm_stats")

    def __call__(self, data: DataDict) -> DataDict:
        if self.norm_stats is None:
            return data

        return apply_tree(
            data,
            self.norm_stats,
            self._unnormalize_quantile if self.use_quantiles else self._unnormalize,
            strict=True,
        )

    def _unnormalize(self, x, stats: NormStats):
        # Check if this is actions and we should use per-timestamp normalization
        if self.use_per_timestamp and stats.per_timestamp_mean is not None and x.ndim >= 2:
            # x has shape [..., action_horizon, action_dim] for actions
            # stats.per_timestamp_mean has shape [action_horizon, action_dim]
            mean = pad_to_dim(stats.per_timestamp_mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.per_timestamp_std, x.shape[-1], axis=-1, value=1.0)
            # Ensure we only use the appropriate timesteps
            mean = mean[: x.shape[-2], :]
            std = std[: x.shape[-2], :]
            return x * (std + 1e-6) + mean
        else:
            # Regular unnormalization
            mean = pad_to_dim(stats.mean, x.shape[-1], axis=-1, value=0.0)
            std = pad_to_dim(stats.std, x.shape[-1], axis=-1, value=1.0)
            return x * (std + 1e-6) + mean

    def _unnormalize_quantile(self, x, stats: NormStats):
        assert stats.q01 is not None
        assert stats.q99 is not None
        
        # Check if this is actions and we should use per-timestamp normalization
        if self.use_per_timestamp and stats.per_timestamp_q01 is not None and x.ndim >= 2:
            # x has shape [..., action_horizon, action_dim] for actions
            # stats.per_timestamp_q01 has shape [action_horizon, action_dim]
            q01 = stats.per_timestamp_q01[: x.shape[-2], : x.shape[-1]]
            q99 = stats.per_timestamp_q99[: x.shape[-2], : x.shape[-1]]
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
        else:
            # Regular quantile unnormalization
            q01, q99 = stats.q01, stats.q99
            if (dim := q01.shape[-1]) < x.shape[-1]:
                return np.concatenate([(x[..., :dim] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01, x[..., dim:]], axis=-1)
            return (x + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


