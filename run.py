import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator

from detectors import FHDSDM
from evaluators import DriftEvaluator
from evaluators.metrics import MaxPerformanceLoss, RestorationTime
from streams import VariableChunkStream, StreamWrapper


def run():
    chunk_size = 1000
    drift_chunk_size = 30
    sl_stream = StreamGenerator(n_chunks=300, chunk_size=chunk_size, n_drifts=5, recurring=True, random_state=42)
    stream = StreamWrapper(sl_stream)
    variable_size_stream = VariableChunkStream(stream)
    clf = MLPClassifier(solver='adam')
    # clf = AUE(GaussianNB())
    # clf = AWE(GaussianNB())
    # clf = OnlineBagging(GaussianNB())
    # clf = SEA(GaussianNB())
    detector = FHDSDM(window_size=1000)
    drift_evaulator = DriftEvaluator(chunk_size, metrics=[RestorationTime(reduction=None), MaxPerformanceLoss(reduction=None)])

    scores, chunk_sizes, drift_indices, stabilization_indices = test_then_train(variable_size_stream, clf, detector, accuracy_score, chunk_size, drift_chunk_size,
                                                                                use_different_chunk_size=False)

    plot_results(scores, chunk_sizes, drift_indices, stabilization_indices)

    restoration_time = RestorationTime(reduction=None)(scores, drift_indices, stabilization_indices)
    max_performance_loss = MaxPerformanceLoss(reduction=None)(scores, drift_indices, stabilization_indices)
    print('restoration_time = ', restoration_time)
    print('max_performance_loss = ', max_performance_loss)

    plt.show()


def test_then_train(stream, clf, detector, metric, chunk_size, drift_chunk_size, use_different_chunk_size=False):
    scores = []
    chunk_sizes = []
    drift_indices = []
    stabilization_indices = []
    drift_phase = False

    for i, (X, y) in enumerate(stream):
        # Test
        if i > 0:
            chunk_sizes.append(X.shape[0])
            y_pred = clf.predict(X)
            score = metric(y, y_pred)
            scores.append(score)
            correct_preds = np.array(y == y_pred, dtype=float)
            detector.add_element(correct_preds)
            if drift_phase and use_different_chunk_size:
                stream.chunk_size = min(int(stream.chunk_size * 1.1), chunk_size)
            if detector.change_detected():
                drift_phase = True
                if use_different_chunk_size:
                    stream.chunk_size = drift_chunk_size
                    detector.batch_size = drift_chunk_size

                print("Change detected, batch:", i)
                drift_indices.append(i)
                # if type(clf) == MLPClassifier:
                #     clf._optimizer.learning_rate = drift_chunk_size / chunk_size * clf._optimizer.learning_rate
            elif detector.stabilization_detected():
                drift_phase = False
                if use_different_chunk_size:
                    stream.chunk_size = chunk_size
                    detector.batch_size = chunk_size
                print("Stabilization detected, batch:", i)
                stabilization_indices.append(i)
                # if type(clf) == MLPClassifier:
                #     clf._optimizer.learning_rate = 0.001
        # Train
        clf.partial_fit(X, y, stream.classes)
        i += 1
    print()

    return np.array(scores), chunk_sizes, drift_indices, stabilization_indices


def plot_results(scores, chunk_sizes, drift_indices, stabilization_indices):
    plt.figure()
    x_sample = np.cumsum(chunk_sizes)
    plt.plot(x_sample, scores, label='accuracy_score')

    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Samples')

    for i in drift_indices:
        plt.axvline(x_sample[i], 0, 1, color='r')
    for i in stabilization_indices:
        plt.axvline(x_sample[i], 0, 1, color='g')


if __name__ == "__main__":
    run()
