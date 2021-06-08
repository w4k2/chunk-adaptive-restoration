import pandas as pd
import numpy as np
import random

from scipy.io import arff
from sklearn import preprocessing
from imblearn.over_sampling import RandomOverSampler


class RecurringInsectsDataset:
    def __init__(self, chunk_size, repetitions=2):
        self._chunk_size = chunk_size
        self._repetitions = repetitions
        self.x, self.y = self.load_insects()
        self._classes = np.unique(self.y)

    def load_insects(self):
        data = arff.loadarff('./streams/repository/INSECTS-incremental-reoccurring_balanced_norm.arff')
        df = pd.DataFrame(data[0])
        le = preprocessing.LabelEncoder()
        df['class'] = le.fit_transform(df['class'])
        data_list = df.values.tolist()
        x = [row[:33] for row in data_list]
        y = [row[-1] for row in data_list]
        # Concept 1
        x1 = x[36000:40000]*15
        y1 = y[36000:40000]*15
        for idx, _ in enumerate(y1):
            if y1[idx] == 5:
                y1[idx] = 4
        shuffle_pack = list(zip(x1, y1))
        random.shuffle(shuffle_pack)
        x1, y1 = zip(*shuffle_pack)
        # Concept 2
        x2 = x[69500:71000]*30
        y2 = y[69500:71000]*30
        shuffle_pack = list(zip(x2, y2))
        random.shuffle(shuffle_pack)
        x2, y2 = zip(*shuffle_pack)
        # Create stream
        x = x1
        x = x + x2
        y = y1 + y2
        for i in range(self._repetitions):
            x = x + x1
            x = x + x2
            y = y + y1 + y2
        x = np.asarray(x, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        return x, y

    def __next__(self):
        i = 0
        while i < self.x.shape[0]:
            x_chunk = self.x[i:i+self._chunk_size]
            y_chunk = self.y[i:i+self._chunk_size]
            ros = RandomOverSampler(random_state=42)
            x_chunk, y_chunk = ros.fit_resample(x_chunk, y_chunk)
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
        concepts_len = 45000+60000
        drift_idx = [60000 - self._chunk_size, concepts_len - self._chunk_size]
        for i in range(1, self._repetitions + 1):
            drift_idx.append(60000 + concepts_len * i - self._chunk_size)
            drift_idx.append(concepts_len + concepts_len * i - self._chunk_size)
        drift_idx.pop()
        print('drift_idx = ', drift_idx)
        return drift_idx
