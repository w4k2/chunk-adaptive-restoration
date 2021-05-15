import scipy.stats
import numpy as np


def main():
    model_names = ['aue', 'awe', 'sea', 'onlinebagging', 'mlp']

    for model_name in model_names:
        metrics_baseline = np.load(f'results/{model_name}_baseline.npy')
        metrics_ours = np.load(f'results/{model_name}_ours.npy')
        stabilization_time_baseline = metrics_baseline[:, 0]
        stabilization_time_ours = metrics_ours[:, 0]
        statistic, p_value = scipy.stats.wilcoxon(stabilization_time_baseline, stabilization_time_ours)
        print(f'\n{model_name}')
        print('statistic = ', statistic)
        print('p_value = ', p_value)


if __name__ == '__main__':
    main()
