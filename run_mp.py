import numpy as np
import math
import argparse
import functools
import multiprocessing
import itertools

from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE

from run import *


def run():
    stream_names = [
        'stream_learn_recurring_abrupt_1',  'stream_learn_recurring_abrupt_2', 'stream_learn_recurring_abrupt_3', 'stream_learn_recurring_abrupt_4',
        'stream_learn_nonrecurring_abrupt_1', 'stream_learn_nonrecurring_abrupt_2', 'stream_learn_nonrecurring_abrupt_3', 'stream_learn_nonrecurring_abrupt_4',
        'stream_learn_recurring_gradual_1', 'stream_learn_recurring_gradual_2', 'stream_learn_recurring_gradual_3', 'stream_learn_recurring_gradual_4',
        'stream_learn_nonrecurring_gradual_1', 'stream_learn_nonrecurring_gradual_2', 'stream_learn_nonrecurring_gradual_3', 'stream_learn_nonrecurring_gradual_4',
        'stream_learn_recurring_incremental_1', 'stream_learn_recurring_incremental_2', 'stream_learn_recurring_incremental_3', 'stream_learn_recurring_incremental_4',
        'stream_learn_nonrecurring_incremental_1', 'stream_learn_nonrecurring_incremental_2', 'stream_learn_nonrecurring_incremental_3', 'stream_learn_nonrecurring_incremental_4',
        'usenet_1',
        'insects_abrupt',
        'insects_gradual',
        # not done svm:
        # 'stream_learn_recurring_gradual_3',
        # 'stream_learn_nonrecurring_gradual_3',
        # 'stream_learn_nonrecurring_incremental_4',
        # 'stream_learn_nonrecurring_abrupt_3',
        # 'stream_learn_nonrecurring_abrupt_4',
    ]

    models_names = ['wae', 'awe', 'sea']
    base_model_name = 'mlp'
    base_models = {
        'naive_bayes': GaussianNB,
        'knn': KNeighborsClassifier,
        'svm': functools.partial(SVC, probability=True),
        'mlp': functools.partial(MLPClassifier, learning_rate_init=0.01),
    }
    metrics_baseline = [[] for _ in models_names]
    metrics_ours = [[] for _ in models_names]

    args = list(itertools.product(stream_names, enumerate(models_names)))

    with multiprocessing.get_context('spawn').Pool(processes=6) as pool:
        worker = functools.partial(get_metric_values, base_model_name=base_model_name, base_models=base_models)
        all_results_list = pool.map(worker, args)

    for model_index, m_baseline, m_ours in all_results_list:
        metrics_baseline[model_index].append(m_baseline)
        metrics_ours[model_index].append(m_ours)

    # for stream_name in stream_names:
    #     for model_index, model_name in enumerate(models_names):
    #         metrics_vales = get_metric_values(stream_name, model_index, model_name, base_model_name, base_models, metrics_baseline, streams_for_plotting, all_axes)
    #         metrics_ours[model_index].append(metrics_vales)

    """
        metrics_baseline and metrics_ours are [NxMx2] tensor (2 is for mean and std)
        N is number of streams
        M is number of metrics,
        order of metrics: SamplewiseStabilizationTime, MaxPerformanceLoss, SamplewiseRestorationTime 0.9, SamplewiseRestorationTime 0.8, SamplewiseRestorationTime 0.7, SamplewiseRestorationTime 0.6
    """

    for i, model_name in enumerate(models_names):
        metrics_b = np.stack(metrics_baseline[i], axis=0)
        metrics_o = np.stack(metrics_ours[i], axis=0)

        np.save(f'results/{model_name}_{base_model_name}_baseline.npy', metrics_b)
        np.save(f'results/{model_name}_{base_model_name}_ours.npy', metrics_o)

def get_metric_values(args, base_model_name=None, base_models=None):
    stream_name, (model_index, model_name) = args
    print(f'\n\n=================={stream_name}================\n\n')
    clf = get_model(model_name, base_models[base_model_name])
    axis = None
    m_baseline = experiment(clf, stream_name, variable_chunk_size=False, axis=axis)
    
    clf = get_model(model_name, base_models[base_model_name])
    m_ours = experiment(clf, stream_name, variable_chunk_size=True, axis=axis)

    print(f'stream = {stream_name}, model index = {model_index}, model_name = {model_name}, m_baseline = {m_baseline}, m_ours = {m_ours}')
    return model_index, m_baseline, m_ours


if __name__ == "__main__":
    run()