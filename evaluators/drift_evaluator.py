import copy
import numpy as np

from detectors import FHDSDM


class DriftEvaluator:

    def __init__(self, batch_size, scores):
        self._fhdsdm = FHDSDM(batch_size=batch_size)
        drift_indices = []
        stabilization_indices = []

        for i, score in enumerate(scores):
            self._fhdsdm.add_element(score)
            if self._fhdsdm.change_detected():
                drift_indices.append(i)
            if self._fhdsdm.stabilization_detected():
                stabilization_indices.append(i)

        self.pairs = self.compute_drift_stabilization_pairs(drift_indices, stabilization_indices)
        self.restoration_times = [stabilization_idx - drift_idx for drift_idx, stabilization_idx in self.pairs]
        self.max_performance_loss_values = self.compute_max_performance_loss(drift_indices, stabilization_indices, scores)

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

    def compute_max_performance_loss(self, drift_indices, stabilization_indices, scores):
        stabilization_indices_copy = copy.deepcopy(stabilization_indices)
        stabilization_indices_copy.insert(0, np.argmax(scores[:drift_indices[0]]))
        max_performance_loss = []
        for i in range(len(stabilization_indices_copy) - 1):
            prev_stabilization_idx = stabilization_indices_copy[i]
            stabilization_idx = stabilization_indices_copy[i+1]
            s_t = min(scores[prev_stabilization_idx], scores[stabilization_idx])
            current_scores = scores[prev_stabilization_idx:stabilization_idx]
            mpl = np.max([(s_t - score)/s_t for score in current_scores])
            max_performance_loss.append(mpl)
        return max_performance_loss

    def _get_reduced(self, metric_name, reduction='avg'):
        """ return computed restoration time.

        Restoration time is computed based on detected drift and stabilization moments.

        Args:
            - reduction - one: 'avg', 'max', 'min' or None.
                if None return sequence of computed restoration times
        Return:
            - restoration time computed with selected reduction method
        """
        metric = getattr(self, metric_name)
        if reduction == 'avg':
            return sum(metric) / len(metric)
        elif reduction == 'max':
            return max(metric)
        elif reduction == 'min':
            return min(metric)
        elif reduction == None:
            return metric
        else:
            raise ValueError(f"Unknown reduction: {rediction}, expected one of ['avg', 'min', 'max']")

    def restoration_time(self, reduction='avg'):
        return self._get_reduced("restoration_times", reduction)

    def max_performance_loss(self, reduction='avg'):
        return self._get_reduced("max_performance_loss_values", reduction)
