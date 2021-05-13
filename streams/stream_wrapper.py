

class StreamWrapper:
    def __init__(self, base_stream):
        self.base_stream = base_stream

    def __next__(self):
        while not self.base_stream.is_dry():
            X, y = self.base_stream.get_chunk()
            yield X, y

    def __iter__(self):
        return next(self)

    @property
    def chunk_size(self):
        return self.base_stream.chunk_size

    @property
    def n_features(self):
        return self.base_stream.n_features

    @property
    def classes(self):
        return self.base_stream.classes_

    @property
    def drift_sample_idx(self):
        if self.base_stream.incremental or self.base_stream.concept_sigmoid_spacing is not None:
            indexes = []
            print(self.base_stream.concept_probabilities)
            print(len(self.base_stream.concept_probabilities))
            for i in range(1, len(self.base_stream.concept_probabilities)):
                if self.base_stream.concept_probabilities[i-1] == 0 and self.base_stream.concept_probabilities[i] != 0:
                    indexes.append(i)
                elif self.base_stream.concept_probabilities[i-1] == 1 and self.base_stream.concept_probabilities[i] != 1:
                    indexes.append(i)
            print('drift sample indexes = ', indexes)
        else:
            stream_len = self.base_stream.n_chunks * self.base_stream.chunk_size
            concept_duration = stream_len // self.base_stream.n_drifts
            indexes = list(range(concept_duration // 2 - self.base_stream.chunk_size, stream_len, concept_duration))
        return indexes
