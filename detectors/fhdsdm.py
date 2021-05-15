import math
import collections
import numpy as np


class FHDSDM:

    def __init__(self, window_size_drift=100, window_size_stabilization=30, delta=0.000001, epsilon_s=0.001):

        self._delta = delta
        self._window_size_drift = window_size_drift
        self._window_drift = collections.deque([], maxlen=window_size_drift)
        self._window_stabilization = collections.deque([], maxlen=window_size_stabilization)
        self._epsilon = math.sqrt(math.log((1 / self._delta), math.e) / (2 * self._window_size_drift))
        self._epsilon_s = epsilon_s
        # print('epsilon_s = ', self._epsilon_s)

        self._p_max = 0
        self._drift_phase = False
        self._drift_started = False
        self._stabilization_phase = False

    def add_element(self, batch_preds):
        """ add accuracy of predictions for one chunk
        """
        self._drift_phase = False
        self._stabilization_phase = False

        acc = sum(batch_preds) / batch_preds.shape[0]
        self._window_stabilization.append(acc)
        if self._drift_started:
            diff = np.array(self._window_stabilization).var()
            # print('stabilization diff = ', diff)
            if diff < self._epsilon_s:
                self._stabilization_phase = True
                self._drift_started = False

        drift_started = False
        for p in batch_preds:
            self._window_drift.append(p)
            if len(self._window_drift) < self._window_size_drift:
                continue
            p_t = sum(self._window_drift) / len(self._window_drift)
            if self._p_max < p_t:
                self._p_max = p_t
            drift_started = drift_started or (self._p_max - p_t) >= self._epsilon

        if drift_started and not self._drift_started:
            self._drift_started = True
            self._drift_phase = True
            self._p_max = 0

    def change_detected(self):
        return self._drift_phase

    def stabilization_detected(self):
        return self._stabilization_phase
