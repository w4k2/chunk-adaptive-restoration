from detectors import FHDSDM


class DriftEvaluator:
    def __init__(self, batch_size, metrics=tuple()):
        self.batch_size = batch_size
        self.metrics = metrics

    def evaluate(self, scores):
        drift_indices = []
        stabilization_indices = []

        fhdsdm = FHDSDM(batch_size=self.batch_size)
        for i, score in enumerate(scores):
            fhdsdm.add_element(score)
            if fhdsdm.change_detected():
                drift_indices.append(i)
            elif fhdsdm.stabilization_detected():
                stabilization_indices.append(i)

        metrics_values = [metric(scores, drift_indices, stabilization_indices) for metric in self.metrics]
        return metrics_values
