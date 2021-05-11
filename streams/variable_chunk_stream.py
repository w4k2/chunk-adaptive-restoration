import numpy as np


class VariableChunkStream:
    def __init__(self, base_stream, chunk_size=None):
        self.base_stream = base_stream
        self.chunk_size = chunk_size
        if self.chunk_size is None:
            self.chunk_size = base_stream.chunk_size

    def __next__(self):
        X_buffer = np.zeros((0, self.base_stream.n_features))
        y_buffer = np.zeros((0,))

        for X, y in self.base_stream:
            X_buffer = np.concatenate((X_buffer, X), axis=0)
            y_buffer = np.concatenate((y_buffer, y), axis=0)
            while X_buffer.shape[0] >= self.chunk_size:
                X_chunk, X_buffer = X_buffer[:self.chunk_size], X_buffer[self.chunk_size:]
                y_chunk, y_buffer = y_buffer[:self.chunk_size], y_buffer[self.chunk_size:]
                yield X_chunk, y_chunk

    def __iter__(self):
        return next(self)

    def change_chunk_size(self, new_size):
        self.chunk_size = new_size

    @property
    def classes(self):
        return self.base_stream.classes
