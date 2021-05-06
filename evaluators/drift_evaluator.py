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

    def make_drift_stabilization_pairs(self, drift_indices, stabilization_indices):
        pass

