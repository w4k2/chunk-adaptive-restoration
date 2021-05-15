from scipy.io import arff
import pandas as pd
import numpy as np
import random


class RecurringUsenetDataset:

    def __init__(self, chunk_size):
        self._repetitions = 2
        self._chunk_size = chunk_size
        self.x, self.y = self.load_usenet()
        self._classes = np.unique(self.y)

    def load_usenet(self):
        data = arff.loadarff('./streams/usenet1.arff')
        df = pd.DataFrame(data[0])

        data_list = df.values.tolist()

        x = [row[:99] for row in data_list]
        y = [row[-1] for row in data_list]

        # Concept 1
        x1 = x[1300:]*100
        y1 = y[1300:]*100

        shuffle_pack = list(zip(x1, y1))
        random.shuffle(shuffle_pack)
        x1, y1 = zip(*shuffle_pack)

        # Concept 2
        x2 = x[400:600]*100
        y2 = y[400:600]*100

        shuffle_pack = list(zip(x2, y2))
        random.shuffle(shuffle_pack)
        x2, y2 = zip(*shuffle_pack)

        x = x1 + x2 + x1 + x2 + x1 + x2
        y = y1 + y2 + y1 + y2 + y1 + y2

        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        return x, y

    def __next__(self):
        i = 0
        while i < self.x.shape[0]:
            x_chunk = self.x[i:i+self._chunk_size]
            y_chunk = self.y[i:i+self._chunk_size]
            yield x_chunk, y_chunk
            i += self._chunk_size

    def __iter__(self):
        return next(self)

    @property
    def chunk_size(self):
        return self._chunk_size

    @property
    def n_features(self):
        return self.x.shape[1]

    @property
    def classes(self):
        return self._classes

    @property
    def drift_sample_idx(self):
        concepts_len = 20000+20000
        drift_idx = [20000 - self._chunk_size, concepts_len - self._chunk_size]
        for i in range(1, self._repetitions + 1):
            drift_idx.append(20000 + concepts_len * i - self._chunk_size)
            drift_idx.append(concepts_len + concepts_len * i - self._chunk_size)
        drift_idx.pop()
        print('drift_idx = ', drift_idx)
        return drift_idx
