from detectors import FHDSDM
from evaluators import DriftEvaluator
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from evaluators.metrics import MaxPerformanceLoss, RestorationTime
import matplotlib.pyplot as plt
import numpy as np


def run():
    chunk_size = 1000
    drift_chunk_size = 100
    stream = StreamGenerator(n_chunks=5000, chunk_size=chunk_size, n_drifts=5, recurring=True, random_state=42)
    clf = MLPClassifier(solver='adam')
    detector = FHDSDM(batch_size=chunk_size)
    drift_evaulator = DriftEvaluator(chunk_size, metrics=[RestorationTime(reduction=None), MaxPerformanceLoss(reduction=None)])

    scores, drift_indices, stabilization_indices = test_then_train(stream, clf, detector, accuracy_score, chunk_size, drift_chunk_size)

    plt.figure()
    plt.plot(scores, label='accuracy_score')

    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Chunk')
    print(scores)

    for i in drift_indices:
        plt.axvline(i, 0, 1, color='r')
    for i in stabilization_indices:
        plt.axvline(i, 0, 1, color='g')

    restoration_time = RestorationTime(reduction=None)(scores, drift_indices, stabilization_indices)
    max_performance_loss = MaxPerformanceLoss(reduction=None)(scores, drift_indices, stabilization_indices)
    print('restoration_time = ', restoration_time)
    print('max_performance_loss = ', max_performance_loss)

    plt.show()


class VariableChunkStream:
    def __init__(self, base_stream, chunk_size=None):
        self.base_stream = base_stream
        self.chunk_size = chunk_size
        if self.chunk_size is None:
            self.chunk_size = base_stream.chunk_size

    def __next__(self):
        X_buffer = np.zeros((0, self.base_stream.n_features))
        y_buffer = np.zeros((0,))
        while not self.base_stream.is_dry():
            X, y = self.base_stream.get_chunk()
            X_buffer = np.concatenate((X_buffer, X), axis=0)
            y_buffer = np.concatenate((y_buffer, y), axis=0)
            while X_buffer.shape[0] >= self.chunk_size:
                X_chunk, X_buffer = X_buffer[:self.chunk_size], X_buffer[self.chunk_size:]
                y_chunk, y_buffer = y_buffer[:self.chunk_size], y_buffer[self.chunk_size:]
                yield X_chunk, y_chunk

    def __iter__(self):
        return next(self)

    def change_chunk_size(self, new_size):
        self.chunk_size = new_size


def test_then_train(stream, clf, detector, metric, chunk_size, drift_chunk_size):
    scores = []
    drift_indices = []
    stabilization_indices = []

    variable_size_stream = VariableChunkStream(stream)
    for i, (X, y) in enumerate(variable_size_stream):
        # Test
        if stream.previous_chunk is not None:
            y_pred = clf.predict(X)
            score = metric(y, y_pred)
            scores.append(score)
            detector.add_element(score)
            if detector.change_detected():
                new_size = max(int(variable_size_stream.chunk_size * 0.5), drift_chunk_size)
                variable_size_stream.chunk_size = new_size
                detector.batch_size = new_size
                # variable_size_stream.chunk_size = drift_chunk_size
                # detector.batch_size = drift_chunk_size
                print("Change detected, batch:", i)
                drift_indices.append(i)
            elif detector.stabilization_detected():
                variable_size_stream.chunk_size = chunk_size
                detector.batch_size = chunk_size
                print("Stabilization detected, batch:", i)
                stabilization_indices.append(i)
        # Train
        clf.partial_fit(X, y, stream.classes_)
        i += 1
    print()

    return np.array(scores), drift_indices, stabilization_indices


if __name__ == "__main__":
    run()
