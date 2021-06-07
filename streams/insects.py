from scipy.io import arff
from sklearn import preprocessing
import pandas as pd
import numpy as np


class InsectsDataset:

    def __init__(self, filename, chunk_size):
        self._chunk_size = chunk_size
        self.filename = filename
        self.x, self.y = self.load_insects(filename)
        self._classes = np.unique(self.y)
        self._classes = np.unique(self.y)

    def load_insects(self, filename):
        data = arff.loadarff(filename)
        df = pd.DataFrame(data[0])

        le = preprocessing.LabelEncoder()
        df['class'] = le.fit_transform(df['class'])

        data_list = df.values.tolist()

        x = [row[:33] for row in data_list]
        y = [row[-1] for row in data_list]

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
        # concepts_len = 20000+20000
        # drift_idx = [20000 - self._chunk_size, concepts_len - self._chunk_size]
        # for i in range(1, self._repetitions + 1):
        #     drift_idx.append(20000 + concepts_len * i - self._chunk_size)
        #     drift_idx.append(concepts_len + concepts_len * i - self._chunk_size)
        # drift_idx.pop()
        # print('drift_idx = ', drift_idx)
        # return drift_idx

        # return [0]

        if self.filename == './streams/insects/INSECTS-abrupt_imbalanced_norm.arff':
            return [83859, 128651, 182320, 242883, 268380]
        elif self.filename == './streams/insects/INSECTS-gradual_imbalanced_norm.arff':
            return [58159]
        elif self.filename == './streams/insects/INSECTS-incremental_imbalanced_norm.arff':
            return [150683, 301365]
