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
    stream = StreamGenerator(n_chunks=5000, chunk_size=chunk_size, n_drifts=5, recurring=True)
    clf = MLPClassifier()
    detector = FHDSDM(batch_size=chunk_size)
    scores = test_then_train(stream, clf, detector, accuracy_score)
    drift_evaulator = DriftEvaluator(chunk_size, metrics=[RestorationTime(reduction=None), MaxPerformanceLoss(reduction=None)])

    plt.figure()
    plt.plot(scores, label='accuracy_score')

    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Chunk')
    print(scores)

    detector = FHDSDM(batch_size=chunk_size)
    for i, score in enumerate(scores):
        detector.add_element(score)
        if detector.change_detected():
            print("Change detected, batch:", i)
            plt.axvline(i, 0, 1, color='r')
        if detector.stabilization_detected():
            print("Stabilization detected, batch:", i)
            plt.axvline(i, 0, 1, color='g')

    restoration_time, max_performance_loss = drift_evaulator.evaluate(scores)
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
        if self.base_stream.is_dry():
            return
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


def test_then_train(stream, clf, detector, metric):
    scores = []

    variable_size_stream = VariableChunkStream(stream)
    i = 0
    for X, y in variable_size_stream:
        # Test
        if stream.previous_chunk is not None:
            y_pred = clf.predict(X)
            score = metric(y, y_pred)
            scores.append(score)
            detector.add_element(score)
            if detector.change_detected():
                variable_size_stream.chunk_size = 200
                detector.batch_size = 200
                print("Change detected, batch:", i)
            elif detector.stabilization_detected():
                variable_size_stream.chunk_size = 1000
                detector.batch_size = 500
                print("Stabilization detected, batch:", i)
        # Train
        clf.partial_fit(X, y, stream.classes_)
        i += 1
    print()

    return np.array(scores)


if __name__ == "__main__":
    run()
