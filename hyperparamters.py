import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
import functools
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.streams import StreamGenerator
from scipy.ndimage.filters import gaussian_filter1d

from detectors import FHDSDM
from evaluators.metrics import MaxPerformanceLoss, SamplewiseStabilizationTime, RestorationTime, SamplewiseRestorationTime
from streams import VariableChunkStream, StreamWrapper, RecurringInsectsDataset, RecurringUsenetDataset, InsectsDataset
from config import configs
from config_real import real_configs
from run import get_model, experiment, test_then_train


def run():
    stream_names = [
        'stream_learn_recurring_abrupt_1', 'stream_learn_recurring_abrupt_2', 'stream_learn_recurring_abrupt_3', 'stream_learn_recurring_abrupt_4',
        'stream_learn_nonrecurring_abrupt_1', 'stream_learn_nonrecurring_abrupt_2', 'stream_learn_nonrecurring_abrupt_3', 'stream_learn_nonrecurring_abrupt_4',
        'stream_learn_recurring_gradual_1', 'stream_learn_recurring_gradual_2', 'stream_learn_recurring_gradual_3', 'stream_learn_recurring_gradual_4',
        'stream_learn_nonrecurring_gradual_1', 'stream_learn_nonrecurring_gradual_2', 'stream_learn_nonrecurring_gradual_3', 'stream_learn_nonrecurring_gradual_4',
        'stream_learn_recurring_incremental_1', 'stream_learn_recurring_incremental_2', 'stream_learn_recurring_incremental_3', 'stream_learn_recurring_incremental_4',
        'stream_learn_nonrecurring_incremental_1', 'stream_learn_nonrecurring_incremental_2', 'stream_learn_nonrecurring_incremental_3', 'stream_learn_nonrecurring_incremental_4',
    ]

    models_names = ['wae', 'awe', 'sea']
    base_model_name = 'naive_bayes'
    base_models = {
        'naive_bayes': GaussianNB,
        'knn': KNeighborsClassifier,
        'svm': functools.partial(SVC, probability=True),
    }

    window_sizes = (30, 50, 100)
    thresholds = (0.1, 0.01, 0.001, 0.0001)

    all_results = [[] for _ in range(len(thresholds))]

    for i, window_size_stabilization in enumerate(window_sizes):
        for stabilization_threshold in thresholds:
            results = []
            for stream_name in stream_names:
                for model_name in models_names:
                    print(f'\n\n=================={stream_name}================\n\n')
                    clf = get_model(model_name, base_models[base_model_name])
                    metrics_vales = experiment(clf, stream_name, variable_chunk_size=True, window_size_stabilization=window_size_stabilization, fhdsdm_epsilon_s=stabilization_threshold)
                    sr08 = metrics_vales[3][0]
                    results.append(sr08)

            avrg_sr = sum(results) / len(results)
            all_results[i].append(avrg_sr)


    all_results_np = np.stack(all_results, axis=0)
    np.save(f'results_hyperparam/sr.npy', all_results_np)

def get_stream(stream_name, cfg):
    if stream_name.startswith('stream_learn'):
        sl_stream = StreamGenerator(
            n_chunks=cfg['n_chunks'],
            chunk_size=cfg['chunk_size'],
            n_drifts=cfg['n_drifts'],
            recurring=cfg['recurring'],
            random_state=42,
            incremental=cfg['incremental'],
            concept_sigmoid_spacing=cfg['concept_sigmoid_spacing'],
        )
        stream = StreamWrapper(sl_stream)
    elif stream_name == 'insects_abrupt':
        stream = InsectsDataset('./streams/insects/INSECTS-abrupt_imbalanced_norm.arff', cfg['chunk_size'])
    elif stream_name == 'insects_gradual':
        stream = InsectsDataset('./streams/insects/INSECTS-gradual_imbalanced_norm.arff', cfg['chunk_size'])
    elif stream_name == 'insects_incremental':
        stream = InsectsDataset('./streams/insects/INSECTS-incremental_imbalanced_norm.arff', cfg['chunk_size'])
    elif stream_name.startswith('usenet'):
        stream = RecurringUsenetDataset(cfg['chunk_size'])
    else:
        raise ValueError(f"Invalid stream name: {stream_name}")
    return stream

if __name__ == "__main__":
    run()