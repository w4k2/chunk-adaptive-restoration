from detectors import FHDSDM
from evaluators import DriftEvaluator
from strlearn.streams import StreamGenerator
from strlearn.evaluators import TestThenTrain
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def run():
    chunk_size = 200
    stream = StreamGenerator(n_chunks=1000, chunk_size=chunk_size, n_drifts=5, recurring=True)
    clf = MLPClassifier()
    metrics = [accuracy_score]
    evaluator = TestThenTrain(metrics)
    detector = FHDSDM(batch_size=chunk_size)

    evaluator.process(stream, clf)

    plt.figure()

    for m, metric in enumerate(metrics):
        plt.plot(evaluator.scores[0, :, m], label=metric.__name__)

    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Chunk')
    print(evaluator.scores)

    for i, score in enumerate(evaluator.scores[0, :, 0]):
        detector.add_element(score)
        if detector.change_detected():
            print("Change detected, batch:", i)
            plt.axvline(i, 0, 1, color='r')
        if detector.stabilization_detected():
            print("Stabilization detected, batch:", i)
            plt.axvline(i, 0, 1, color='g')

    drift_evaulator = DriftEvaluator(chunk_size, evaluator.scores[0, :, 0])
    print('restoration_time = ', drift_evaulator.restoration_time(reduction=None))
    print('max_performance_loss = ', drift_evaulator.max_performance_loss(reduction=None))

    plt.show()


if __name__ == "__main__":
    run()
