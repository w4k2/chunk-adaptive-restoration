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

    def __call__(self, scores, chunk_sizes, drift_indices, stabilization_indices):
        values = self.compute(scores, chunk_sizes, drift_indices, stabilization_indices)
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

    def compute(self, scores, chunk_sizes, drift_indices, stabilization_indices):
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


class RestorationTime(StreamMetric, DriftStabilizationMixIn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, scores, chunk_sizes, drift_indices, stabilization_indices):
        pairs = self.compute_drift_stabilization_pairs(drift_indices, stabilization_indices)
        values = [stabilization_idx - drift_idx for drift_idx, stabilization_idx in pairs]
        return values


class SamplewiseRestorationTime(StreamMetric, DriftStabilizationMixIn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, scores, chunk_sizes, drift_indices, stabilization_indices):
        pairs = self.compute_drift_stabilization_pairs(drift_indices, stabilization_indices)
        values = [sum(chunk_sizes[drift_idx:stabilization_idx+1]) for drift_idx, stabilization_idx in pairs]
        return values
