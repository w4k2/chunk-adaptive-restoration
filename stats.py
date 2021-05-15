import scipy.stats
import numpy as np


def main():
    model_names = ['aue', 'awe', 'sea', 'onlinebagging', 'mlp']

    metrics_names = ['SamplewiseStabilizationTime', 'MaxPerformanceLoss', 'SamplewiseRestorationTime0.9',
                     'SamplewiseRestorationTime0.8', 'SamplewiseRestorationTime0.7', 'SamplewiseRestorationTime0.6']

    """
        metrics_baseline and metrics_ours are [NxM] matrixes
        N is number of streams
        M is number of metrics, 
        order of metrics: SamplewiseStabilizationTime, MaxPerformanceLoss, SamplewiseRestorationTime 0.9, SamplewiseRestorationTime 0.8, SamplewiseRestorationTime 0.7, SamplewiseRestorationTime 0.6
    """
    for model_name in model_names:
        print(f'===================model {model_name}===================')
        metrics_baseline = np.load(f'results/{model_name}_baseline.npy')
        metrics_ours = np.load(f'results/{model_name}_ours.npy')
        for i, metric_name in enumerate(metrics_names):
            print(metric_name)
            stabilization_time_baseline = metrics_baseline[:, i]
            stabilization_time_ours = metrics_ours[:, i]
            statistic, p_value = scipy.stats.wilcoxon(stabilization_time_baseline, stabilization_time_ours)
            print('statistic = ', statistic)
            print('p_value = ', p_value)


if __name__ == '__main__':
    main()
