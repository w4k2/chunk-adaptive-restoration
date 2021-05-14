import abc
import copy
import numpy as np


class StreamMetric(abc.ABC):
    def __init__(self, reduction='avg'):
        """ Metric base class
        Args:
            - reduction - one: 'avg', 'max', 'min' or None.
                if None return sequence of computed restoration times
        """
        self.reduction = reduction

    def __call__(self, scores, chunk_sizes, drift_sample_idx, drift_indices, stabilization_indices):
        values = self.compute(scores, chunk_sizes, drift_sample_idx, drift_indices, stabilization_indices)
        return self.reduce(values)

    @abc.abstractmethod
    def compute(self, scores, drift_indices, stabilization_indices):
        raise NotImplementedError

    def reduce(self, values):
        if self.reduction == 'avg':
            return sum(values) / len(values)
        elif self.reduction == 'max':
            return max(values)
        elif self.reduction == 'min':
            return min(values)
        elif self.reduction == None:
            return values
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}, expected one of ['avg', 'min', 'max', None]")


class MaxPerformanceLoss(StreamMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, scores, chunk_sizes, drift_sample_idx, drift_indices, stabilization_indices):
        if len(drift_indices) == 0:
            return []
        stb_indices = copy.deepcopy(stabilization_indices)
        stb_indices.insert(0, np.argmax(scores[:drift_indices[0]]))
        values = []
        for i in range(len(stb_indices) - 1):
            prev_stb_idx = stb_indices[i]
            stb_idx = stb_indices[i+1]
            s_t = min(scores[prev_stb_idx], scores[stb_idx])
            current_scores = scores[prev_stb_idx:stb_idx]
            mpl = max((s_t - score)/s_t for score in current_scores)
            values.append(mpl)
        return values


class DriftStabilizationMixIn:
    def compute_drift_stabilization_pairs(self, drift_indices, stabilization_indices):
        pairs = []
        for i in range(len(drift_indices)-1):
            for j in range(len(stabilization_indices)):
                if stabilization_indices[j] > drift_indices[i] and stabilization_indices[j] < drift_indices[i+1]:
                    pairs.append((drift_indices[i], stabilization_indices[j]))
                    break
        for j in range(len(stabilization_indices)):
            if stabilization_indices[j] > drift_indices[-1]:
                pairs.append((drift_indices[-1], stabilization_indices[j]))
                break
        return pairs


class StabilizationTime(StreamMetric, DriftStabilizationMixIn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, scores, chunk_sizes, drift_sample_idx, drift_indices, stabilization_indices):
        pairs = self.compute_drift_stabilization_pairs(drift_indices, stabilization_indices)
        values = [stabilization_idx - drift_idx for drift_idx, stabilization_idx in pairs]
        return values


class SamplewiseStabilizationTime(StreamMetric, DriftStabilizationMixIn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, scores, chunk_sizes, drift_sample_idx, drift_indices, stabilization_indices):
        pairs = self.compute_drift_stabilization_pairs(drift_indices, stabilization_indices)
        values = [sum(chunk_sizes[drift_idx:stabilization_idx+1]) for drift_idx, stabilization_idx in pairs]
        return values


class RestorationTime(StreamMetric):
    def __init__(self, *args, percentage=0.8, **kwargs):
        super().__init__(*args, **kwargs)
        self._percentage = percentage
        assert 0.0 < percentage <= 1.0, "percentage should be in (0.0, 1.0] interval"

    def compute(self, scores, chunk_sizes, drift_sample_idx, drift_indices, stabilization_indices):
        values = []
        for i, idx in enumerate(drift_indices):
            score_before_drift = scores[idx-1-5]  # -5 is only for debugging
            restoration_threshold = self._percentage * score_before_drift
            next_drift_border = drift_indices[i+1] if i < len(drift_indices)-1 else len(scores)
            restoration_time = None
            for j in range(idx+1, next_drift_border):
                if scores[j] >= restoration_threshold:
                    restoration_time = j - idx
                    break
            values.append(restoration_time)
        return values


class SamplewiseRestorationTime(StreamMetric):
    def __init__(self, percentage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._percentage = percentage
        assert 0.0 < percentage <= 1.0, "percentage should be in (0.0, 1.0] interval"

    def compute(self, scores, chunk_sizes, drift_sample_idx, drift_indices, stabilization_indices):
        chunk_sizes_cumsum = np.cumsum(chunk_sizes)
        drift_chunk_idx = [np.argmax(chunk_sizes_cumsum > sample_idx) for sample_idx in drift_sample_idx]
        # print('drift_chunk_idx = ', drift_chunk_idx)
        values = []
        for i, idx in enumerate(drift_chunk_idx):
            next_drift_border = drift_chunk_idx[i+1] if i < len(drift_chunk_idx)-1 else len(scores)
            min_idx = np.argmin(scores[idx:next_drift_border]) + idx
            # print('min_idx = ', min_idx)
            # print('min_score = ', scores[min_idx])
            max_score = max(scores[min_idx:next_drift_border])
            # print('max_score = ', max_score)
            restoration_threshold = self._percentage * max_score
            restoration_time = None
            for j in range(min_idx, next_drift_border):
                # print('scores[j] = ', scores[j])
                if scores[j] >= restoration_threshold:
                    restoration_time = sum(chunk_sizes[idx:j+1])
                    break
            values.append(restoration_time)
        return values
