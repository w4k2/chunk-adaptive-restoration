

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
        stream_len = self.base_stream.n_chunks * self.base_stream.chunk_size
        concept_duration = stream_len // self.base_stream.n_drifts
        return list(range(concept_duration // 2 - self.base_stream.chunk_size, stream_len, concept_duration))
