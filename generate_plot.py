import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
from sklearn.naive_bayes import GaussianNB
from strlearn.ensembles import AUE, AWE, OnlineBagging, SEA, WAE
from strlearn.streams import StreamGenerator
from streams import VariableChunkStream, StreamWrapper, RecurringInsectsDataset, RecurringUsenetDataset, InsectsDataset
from detectors import FHDSDM


from run import test_then_train


def plot_results(axis, scores, chunk_sizes, drift_sample_idx, drift_detections_idx, stabilization_idx):
    x_sample = np.cumsum(chunk_sizes)
    scores_smooth = gaussian_filter1d(scores, sigma=1)
    # scores_smooth = scores
    axis.plot(x_sample, scores_smooth)
    # axis.legend()

    axis.set_ylim(0, 1)
    axis.set_ylabel('Accuracy')
    axis.set_xlabel('Samples')

    for i in drift_sample_idx:
        axis.axvline(i, 0, 1, color='c')
    for i in drift_detections_idx:
        axis.axvline(x_sample[i], 0, 1, color='r')
    for i in stabilization_idx:
        axis.axvline(x_sample[i], 0, 1, color='g')


model = AUE(GaussianNB())

sl_stream = StreamGenerator(
    n_chunks=100,
    chunk_size=100,
    n_drifts=1,
    # recurring=cfg['recurring'],
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

scores, chunk_sizes, drift_indices, stabilization_indices = test_then_train(stream, model, detector, 100, 100)

plt.figure()
axis = plt.gca()
plot_results(axis, scores, chunk_sizes, stream.drift_sample_idx, drift_indices, stabilization_indices)
plt.show()
