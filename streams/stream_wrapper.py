

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
            fall = self.base_stream.concept_probabilities > 0.99
            for i in range(1, len(fall)):
                if fall[i-1] == True and fall[i] == False:
                    indexes.append(i)
            rise = self.base_stream.concept_probabilities < 0.01
            for i in range(1, len(rise)):
                if rise[i-1] == True and rise[i] == False:
                    indexes.append(i)
            indexes = sorted(indexes)
        else:
            stream_len = self.base_stream.n_chunks * self.base_stream.chunk_size
            concept_duration = stream_len // self.base_stream.n_drifts
            indexes = list(range(concept_duration // 2 - self.base_stream.chunk_size, stream_len, concept_duration))
        return indexes
