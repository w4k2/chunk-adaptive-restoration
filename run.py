import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.evaluators import TestThenTrain
from strlearn.streams import StreamGenerator
from scipy.ndimage.filters import gaussian_filter1d

from detectors import FHDSDM
from evaluators.metrics import MaxPerformanceLoss, SamplewiseStabilizationTime, RestorationTime, SamplewiseRestorationTime
from streams import VariableChunkStream, StreamWrapper, RecurringInsectsDataset
from config import configs


def run():
    stream_names = ['stream_learn_recurring_abrupt_3', ]  # 'insects_3']

    models = [MLPClassifier(learning_rate_init=0.01), AUE(GaussianNB()), SEA(GaussianNB()), AWE(GaussianNB()), OnlineBagging(GaussianNB()), ]
    for stream_name in stream_names:
        for model in models:
            for variable_chunk_size in [False, True]:
                clf = copy.deepcopy(model)
                experiment(clf, stream_name, variable_chunk_size=variable_chunk_size)
            break


def get_stream(stream_name, cfg):
    if stream_name.startswith('stream_learn'):
        sl_stream = StreamGenerator(
            n_chunks=cfg['n_chunks'],
            chunk_size=cfg['chunk_size'],
            n_drifts=cfg['n_drifts'],
            recurring=cfg['recurring'],
            random_state=cfg['random_state'],
            incremental=cfg['incremental'],
            concept_sigmoid_spacing=cfg['concept_sigmoid_spacing'],
        )
        stream = StreamWrapper(sl_stream)
    elif stream_name.startswith('insects'):
        stream = RecurringInsectsDataset(cfg['chunk_size'], repetitions=cfg['n_drifts'])
    else:
        raise ValueError(f"Invalid stream name: {stream_name}")
    return stream


def experiment(clf, stream_name, variable_chunk_size=False):
    cfg = configs[stream_name]
    stream = get_stream(stream_name, cfg)
    variable_size_stream = VariableChunkStream(stream)
    detector = FHDSDM(
        window_size_drift=cfg['fhdsdm_window_size_drift'],
        window_size_stabilization=cfg['fhdsdm_window_size_stabilization'],
        epsilon_s=cfg['fhdsdm_epsilon_s']
    )
    scores, chunk_sizes, drift_indices, stabilization_indices = test_then_train(variable_size_stream, clf, detector, accuracy_score, cfg['chunk_size'], cfg['drift_chunk_size'],
                                                                                variable_chunk_size=variable_chunk_size)

    plot_results(scores, chunk_sizes, stream.drift_sample_idx, drift_indices, stabilization_indices)
    plt.savefig(f'plots/stream_{stream_name}_classifer_{clf.__class__.__name__}_variable_chunk_size_{variable_chunk_size}.png')

    metrics = [
        SamplewiseStabilizationTime(reduction=None),
        MaxPerformanceLoss(reduction=None),
        SamplewiseRestorationTime(percentage=0.9, reduction=None),
        SamplewiseRestorationTime(percentage=0.8, reduction=None),
        SamplewiseRestorationTime(percentage=0.7, reduction=None),
        SamplewiseRestorationTime(percentage=0.6, reduction=None),
    ]
    metrics_vales = [metric(scores, chunk_sizes, stream.drift_sample_idx, drift_indices, stabilization_indices) for metric in metrics]
    print('stabilization_time = ', metrics_vales[0])
    print('max_performance_loss = ', metrics_vales[1])
    print('restoration_time 0.9 = ', metrics_vales[2])
    print('restoration_time 0.8 = ', metrics_vales[3])
    print('restoration_time 0.7 = ', metrics_vales[4])
    print('restoration_time 0.6 = ', metrics_vales[5])
    print('restoration_time 0.5 = ', metrics_vales[6])


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
                if type(clf) == MLPClassifier:
                    clf._optimizer.learning_rate = math.sqrt(stream.chunk_size / chunk_size) * clf._optimizer.learning_rate
            if detector.change_detected():
                drift_phase = True
                if variable_chunk_size:
                    stream.chunk_size = drift_chunk_size
                    detector.batch_size = drift_chunk_size
                    if type(clf) == MLPClassifier:
                        clf._optimizer.learning_rate = math.sqrt(drift_chunk_size / chunk_size) * clf._optimizer.learning_rate
                print("Change detected, batch:", i)
                drift_indices.append(i)
            elif detector.stabilization_detected():
                drift_phase = False
                if variable_chunk_size:
                    stream.chunk_size = chunk_size
                    detector.batch_size = chunk_size
                    if type(clf) == MLPClassifier:
                        clf._optimizer.learning_rate = 0.01
                print("Stabilization detected, batch:", i)
                stabilization_indices.append(i)
        # Train
        clf.partial_fit(X, y, stream.classes)
        i += 1
    print()

    return np.array(scores), chunk_sizes, drift_indices, stabilization_indices


def plot_results(scores, chunk_sizes, drift_sample_idx, drift_detections_idx, stabilization_idx):
    plt.figure(figsize=(22, 12))
    x_sample = np.cumsum(chunk_sizes)
    # scores_smooth = gaussian_filter1d(scores, sigma=1)
    scores_smooth = scores
    plt.plot(x_sample, scores_smooth, label='accuracy_score')

    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.xlabel('Samples')

    for i in drift_sample_idx:
        plt.axvline(i, 0, 1, color='c')
    for i in drift_detections_idx:
        plt.axvline(x_sample[i], 0, 1, color='r')
    for i in stabilization_idx:
        plt.axvline(x_sample[i], 0, 1, color='g')


if __name__ == "__main__":
    run()
