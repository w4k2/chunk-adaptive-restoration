import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.naive_bayes import GaussianNB
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.streams import StreamGenerator
from streams import VariableChunkStream, StreamWrapper, RecurringInsectsDataset, RecurringUsenetDataset, InsectsDataset
from detectors import FHDSDM


from run import test_then_train


def main():
    plt.figure()
    axis = plt.gca()

    axis.set_ybound(upper=1)
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Chunks')

    chunk_sizes = (100, 1000, 10000, 100000)
    for chunk_size in chunk_sizes:
        model = AWE(GaussianNB(), n_estimators=5)

        sl_stream = StreamGenerator(
            n_chunks=100,
            chunk_size=chunk_size,
            n_drifts=1,
            recurring=True,
            random_state=42,
            incremental=False,
            # concept_sigmoid_spacing=cfg['concept_sigmoid_spacing'],
        )
        stream = StreamWrapper(sl_stream)
        detector = FHDSDM(
            window_size_drift=1000,
            window_size_stabilization=30,
            epsilon_s=0.001
        )

        scores, chunk_sizes, drift_indices, stabilization_indices = test_then_train(stream, model, detector, chunk_size, chunk_size)

        # scores = gaussian_filter1d(scores, sigma=1)
        axis.plot(scores, label=f'{chunk_size}')
    axis.legend()
    plt.show()


if __name__ == '__main__':
    main()
