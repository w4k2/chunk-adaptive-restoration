import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator
from scipy.ndimage.filters import gaussian_filter1d

from detectors import FHDSDM
from evaluators.metrics import MaxPerformanceLoss, SamplewiseRestorationTime
from streams import VariableChunkStream, StreamWrapper, RecurringInsectsDataset


configs = {
    'stream_learn': {
        'chunk_size': 1000,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': True,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
    },
    'insects': {
        'chunk_size': 50,
        'drift_chunk_size': 30,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
    }
}


def run():
    stream_name = 'insects'
    cfg = configs[stream_name]

    models = [MLPClassifier(), AUE(GaussianNB()), AWE(GaussianNB()), OnlineBagging(GaussianNB()), SEA(GaussianNB())]
    seeds = [1, ]  # 2, 4, 5, 9]  # 1, 2, 4, 5, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 1415, 1418, 1420, 1421, 1422, 1430, 1433, 1435, 1439, 1440, 1442, 1444, 1467
    for clf in models:
        for seed in seeds:
            for variable_chunk_size in [False, True]:
                stream = get_stream(stream_name, cfg, random_state=seed)
                experiment(clf, stream, cfg, variable_chunk_size=variable_chunk_size)
        break


def get_stream(stream_name, cfg, random_state=42):
    if stream_name == 'stream_learn':
        sl_stream = StreamGenerator(n_chunks=cfg['n_chunks'], chunk_size=cfg['chunk_size'], n_drifts=cfg['n_drifts'], recurring=cfg['recurring'], random_state=random_state)
        stream = StreamWrapper(sl_stream)
    elif stream_name == 'insects':
        stream = RecurringInsectsDataset(cfg['chunk_size'])
    else:
        raise ValueError(f"Invalid stream name: {stream_name}")
    return stream


def experiment(clf, stream, cfg, variable_chunk_size=False):
    variable_size_stream = VariableChunkStream(stream)
    detector = FHDSDM(
        window_size_drift=cfg['fhdsdm_window_size_drift'],
        window_size_stabilization=cfg['fhdsdm_window_size_stabilization'],
        epsilon_s=cfg['fhdsdm_epsilon_s']
    )
    scores, chunk_sizes, drift_indices, stabilization_indices = test_then_train(variable_size_stream, clf, detector, accuracy_score, cfg['chunk_size'], cfg['drift_chunk_size'],
                                                                                variable_chunk_size=variable_chunk_size)

    plot_results(scores, chunk_sizes, drift_indices, stabilization_indices)
    plt.savefig(f'plots/classifer_{clf.__class__.__name__}_variable_chunk_size_{variable_chunk_size}.png')

    restoration_time = SamplewiseRestorationTime(reduction=None)(scores, chunk_sizes, drift_indices, stabilization_indices)
    max_performance_loss = MaxPerformanceLoss(reduction=None)(scores, chunk_sizes, drift_indices, stabilization_indices)
    print('restoration_time = ', restoration_time)
    print('max_performance_loss = ', max_performance_loss)


def test_then_train(stream, clf, detector, metric, chunk_size, drift_chunk_size, variable_chunk_size=False):
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
            if drift_phase and variable_chunk_size:
                stream.chunk_size = min(int(stream.chunk_size * 1.1), chunk_size)
            if detector.change_detected():
                drift_phase = True
                if variable_chunk_size:
                    stream.chunk_size = drift_chunk_size
                    detector.batch_size = drift_chunk_size

                print("Change detected, batch:", i)
                drift_indices.append(i)
                # if type(clf) == MLPClassifier:
                #     clf._optimizer.learning_rate = drift_chunk_size / chunk_size * clf._optimizer.learning_rate
            elif detector.stabilization_detected():
                drift_phase = False
                if variable_chunk_size:
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
    plt.figure(figsize=(22, 12))
    x_sample = np.cumsum(chunk_sizes)
    scores_smooth = gaussian_filter1d(scores, sigma=1)
    plt.plot(x_sample, scores_smooth, label='accuracy_score')

    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Samples')

    for i in drift_indices:
        plt.axvline(x_sample[i], 0, 1, color='r')
    for i in stabilization_indices:
        plt.axvline(x_sample[i], 0, 1, color='g')


if __name__ == "__main__":
    run()
