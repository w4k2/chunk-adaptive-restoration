import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.naive_bayes import GaussianNB
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.streams import StreamGenerator
from streams import VariableChunkStream, StreamWrapper, RecurringInsectsDataset, RecurringUsenetDataset, InsectsDataset
from detectors import FHDSDM


from run import experiment, test_then_train


class NoisyStream:
    def __init__(self, stream_wrapper, noise_rate=0.1):
        self.stream_wrapper = stream_wrapper
        self.noise_rate = noise_rate

    def __iter__(self):
        for X, y in self.stream_wrapper:
            noise_idx = np.argwhere(np.random.binomial(1, self.noise_rate, size=y.shape) == 1)
            y[noise_idx] = - (y[noise_idx] - 1)
            yield X, y

    @property
    def chunk_size(self):
        return self.stream_wrapper.chunk_size

    @chunk_size.setter
    def chunk_size(self, size):
        self.stream_wrapper.chunk_size = size

    @property
    def n_features(self):
        return self.stream_wrapper.n_features

    @property
    def classes(self):
        return self.stream_wrapper.classes

    @property
    def drift_sample_idx(self):
        if self.stream_wrapper.incremental or self.stream_wrapper.concept_sigmoid_spacing is not None:
            indexes = []
            fall = self.stream_wrapper.concept_probabilities > 0.99
            for i in range(1, len(fall)):
                if fall[i-1] == True and fall[i] == False:
                    indexes.append(i)
            rise = self.stream_wrapper.concept_probabilities < 0.01
            for i in range(1, len(rise)):
                if rise[i-1] == True and rise[i] == False:
                    indexes.append(i)
            indexes = sorted(indexes)
        else:
            stream_len = self.stream_wrapper.n_chunks * self.stream_wrapper.chunk_size
            concept_duration = stream_len // self.stream_wrapper.n_drifts
            indexes = list(range(concept_duration // 2 - self.stream_wrapper.chunk_size, stream_len, concept_duration))
        return indexes


def main():
    _, axs = plt.subplots(2)

    noise_rates = (0.0, 0.1, 0.2, 0.3, 0.4)
    stabilization_epsilons = (0.001, 0.001, 0.0, 0.0, 0.0)

    for noise_rate, epsilon in zip(noise_rates, stabilization_epsilons):
        scores, chunk_sizes = experiment(noise_rate, variable_chunk_size=False, epsilon=epsilon)
        plot_results(axs[0], scores, chunk_sizes, label=f'{noise_rate}')

    for noise_rate, epsilon in zip(noise_rates, stabilization_epsilons):
        scores, chunk_sizes = experiment(noise_rate, variable_chunk_size=True, epsilon=epsilon)
        plot_results(axs[1], scores, chunk_sizes, label=f'{noise_rate}')

    axs[0].legend()
    axs[1].legend()
    plt.show()


def experiment(noise_rate, variable_chunk_size=False, epsilon=0.001):
    model = AUE(GaussianNB())

    sl_stream = StreamGenerator(
        n_chunks=100,
        chunk_size=1000,
        n_drifts=1,
        random_state=1410,
        incremental=False,
    )
    wrapper = StreamWrapper(sl_stream)
    variable_size_stream = VariableChunkStream(wrapper)
    stream = NoisyStream(variable_size_stream, noise_rate=noise_rate)
    detector = FHDSDM(
        window_size_drift=1000,
        window_size_stabilization=30,
        epsilon_s=epsilon
    )

    scores, chunk_sizes, _, _ = test_then_train(stream, model, detector, 1000, 100, variable_chunk_size=variable_chunk_size)
    return scores, chunk_sizes


def plot_results(axis, scores, chunk_sizes, label=None):
    x_sample = np.cumsum(chunk_sizes)
    axis.plot(x_sample, scores, label=label)

    axis.set_ylim(0, 1)
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Samples')


if __name__ == '__main__':
    main()
