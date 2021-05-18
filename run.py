import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.streams import StreamGenerator
from scipy.ndimage.filters import gaussian_filter1d

from detectors import FHDSDM
from evaluators.metrics import MaxPerformanceLoss, SamplewiseStabilizationTime, RestorationTime, SamplewiseRestorationTime
from streams import VariableChunkStream, StreamWrapper, RecurringInsectsDataset, RecurringUsenetDataset
from config import configs
from config_real import real_configs


def run():
    args = parse_args()
    stream_names = [
        'stream_learn_recurring_abrupt_1',  'stream_learn_recurring_abrupt_2', 'stream_learn_recurring_abrupt_3', 'stream_learn_recurring_abrupt_4',
        'stream_learn_nonrecurring_abrupt_1',  'stream_learn_nonrecurring_abrupt_2', 'stream_learn_nonrecurring_abrupt_3', 'stream_learn_nonrecurring_abrupt_4',
        'stream_learn_recurring_gradual_1', 'stream_learn_recurring_gradual_2', 'stream_learn_recurring_gradual_3', 'stream_learn_recurring_gradual_4',
        'stream_learn_nonrecurring_gradual_1', 'stream_learn_nonrecurring_gradual_2', 'stream_learn_nonrecurring_gradual_3', 'stream_learn_nonrecurring_gradual_4',
        'stream_learn_recurring_incremental_1', 'stream_learn_recurring_incremental_2', 'stream_learn_recurring_incremental_3', 'stream_learn_recurring_incremental_4',
        'stream_learn_nonrecurring_incremental_1', 'stream_learn_nonrecurring_incremental_2', 'stream_learn_nonrecurring_incremental_3', 'stream_learn_nonrecurring_incremental_4',
        'usenet_1',
    ]

    metrics_baseline = []
    metrics_ours = []
    fig, axes = plt.subplots(3, 2)
    fig.set_size_inches(18.5, 10.5)
    axis_row = 0
    for stream_name in stream_names:
        print(f'\n\n=================={stream_name}================\n\n')
        clf = get_model(args.model_name)
        if stream_name[-1:] == '1' and 'nonrecurring' in stream_name:
            axis = axes[axis_row][0]
        else:
            axis = None
        metrics_vales = experiment(clf, stream_name, variable_chunk_size=False, axis=axis)
        metrics_baseline.append(metrics_vales)

        clf = get_model(args.model_name)
        if axis:
            axis = axes[axis_row][1]
            axis_row += 1
        metrics_vales = experiment(clf, stream_name, variable_chunk_size=True, axis=axis)
        metrics_ours.append(metrics_vales)

    fig.savefig(f'plots/classifer_{clf.__class__.__name__}.png')

    """
        metrics_baseline and metrics_ours are [NxM] matrixes
        N is number of streams
        M is number of metrics, 
        order of metrics: SamplewiseStabilizationTime, MaxPerformanceLoss, SamplewiseRestorationTime 0.9, SamplewiseRestorationTime 0.8, SamplewiseRestorationTime 0.7, SamplewiseRestorationTime 0.6
    """
    metrics_baseline = np.stack(metrics_baseline, axis=0)
    metrics_ours = np.stack(metrics_ours, axis=0)

    np.save(f'results/{args.model_name}_baseline.npy', metrics_baseline)
    np.save(f'results/{args.model_name}_ours.npy', metrics_ours)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', choices=['aue', 'awe', 'sea', 'onlinebagging', 'mlp'], required=True)

    args = parser.parse_args()
    return args


def get_model(model_name):
    models = {
        'aue': AUE(GaussianNB()),
        'awe': AWE(GaussianNB()),
        'sea': SEA(GaussianNB()),
        'onlinebagging': OnlineBagging(GaussianNB()),
        'mlp': MLPClassifier(learning_rate_init=0.01),
    }
    return models[model_name]


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
    elif stream_name.startswith('usenet'):
        stream = RecurringUsenetDataset(cfg['chunk_size'])
    else:
        raise ValueError(f"Invalid stream name: {stream_name}")
    return stream


def experiment(clf, stream_name, variable_chunk_size=False, axis=None):
    cfg = configs[stream_name]
    stream = get_stream(stream_name, cfg)
    variable_size_stream = VariableChunkStream(stream)
    detector = FHDSDM(
        window_size_drift=cfg['fhdsdm_window_size_drift'],
        window_size_stabilization=cfg['fhdsdm_window_size_stabilization'],
        epsilon_s=cfg['fhdsdm_epsilon_s']
    )
    scores, chunk_sizes, drift_indices, stabilization_indices = test_then_train(variable_size_stream, clf, detector, cfg['chunk_size'], cfg['drift_chunk_size'],
                                                                                variable_chunk_size=variable_chunk_size)

    if axis:
        plot_results(axis, scores, chunk_sizes, stream.drift_sample_idx, drift_indices, stabilization_indices)

    metrics = [
        SamplewiseStabilizationTime(reduction='avg'),
        MaxPerformanceLoss(reduction='avg'),
        SamplewiseRestorationTime(percentage=0.9, reduction='avg'),
        SamplewiseRestorationTime(percentage=0.8, reduction='avg'),
        SamplewiseRestorationTime(percentage=0.7, reduction='avg'),
        SamplewiseRestorationTime(percentage=0.6, reduction='avg'),
    ]
    restoration_time_metrics = [
        SamplewiseRestorationTime(percentage=0.9, reduction=None),
        SamplewiseRestorationTime(percentage=0.8, reduction=None),
        SamplewiseRestorationTime(percentage=0.7, reduction=None),
        SamplewiseRestorationTime(percentage=0.6, reduction=None),
    ]
    metrics_vales = [metric(scores, chunk_sizes, stream.drift_sample_idx, drift_indices, stabilization_indices) for metric in metrics]
    print('stabilization_time = ', metrics_vales[0])
    print('max_performance_loss = ', metrics_vales[1])
    for m in restoration_time_metrics:
        restoration_time = m(scores, chunk_sizes, stream.drift_sample_idx, drift_indices, stabilization_indices)
        print(f'restoration_time {m._percentage} = ', restoration_time)
    print('avg restoration_time 0.9 = ', metrics_vales[2])
    print('avg restoration_time 0.8 = ', metrics_vales[3])
    print('avg restoration_time 0.7 = ', metrics_vales[4])
    print('avg restoration_time 0.6 = ', metrics_vales[5])
    return np.array(metrics_vales)


def test_then_train(stream, clf, detector, chunk_size, drift_chunk_size, variable_chunk_size=False):
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
            scores.append(accuracy_score(y, y_pred))
            correct_preds = np.array(y == y_pred, dtype=float)
            detector.add_element(correct_preds)
            if drift_phase and variable_chunk_size:
                stream.chunk_size = min(int(stream.chunk_size * 1.1), chunk_size)
            if detector.change_detected():
                drift_phase = True
                if variable_chunk_size:
                    stream.chunk_size = drift_chunk_size
                print("Change detected, batch:", i)
                drift_indices.append(i-1)
            elif detector.stabilization_detected():
                drift_phase = False
                if variable_chunk_size:
                    stream.chunk_size = chunk_size
                print("Stabilization detected, batch:", i)
                stabilization_indices.append(i-1)
        # Train
        clf.partial_fit(X, y, stream.classes)
    print()

    return np.array(scores), chunk_sizes, drift_indices, stabilization_indices


def plot_results(axis, scores, chunk_sizes, drift_sample_idx, drift_detections_idx, stabilization_idx):
    x_sample = np.cumsum(chunk_sizes)
    scores_smooth = gaussian_filter1d(scores, sigma=1)
    # scores_smooth = scores
    axis.plot(x_sample, scores_smooth, label='accuracy_score')

    axis.set_ylim(0, 1)
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Samples')

    for i in drift_sample_idx:
        axis.axvline(i, 0, 1, color='c')
    for i in drift_detections_idx:
        axis.axvline(x_sample[i], 0, 1, color='r')
    for i in stabilization_idx:
        axis.axvline(x_sample[i], 0, 1, color='g')


if __name__ == "__main__":
    run()
