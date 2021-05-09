import math
import collections


class FHDSDM:

    def __init__(self, batch_size=100, delta=0.000001):

        self._delta = delta
        self._window_size = batch_size
        self._epsilon = math.sqrt(math.log((1 / self._delta), math.e) / (2 * self._window_size))
        self._epsilon_s = self._epsilon
        print('epislon_s = ', self._epsilon_s)

        self._p_max = 0
        self._drift_phase = False
        self._drift_started = False
        self._stabilization_phase = False
        self._batch_counter = 0
        self._stabilization_threshold = 20
        self._window = collections.deque([], maxlen=10)

    def add_element(self, p_t):
        """ add accuracy of predictions for one chunk
        """
        self._drift_phase = False
        self._stabilization_phase = False

        if self._drift_started:
            # diff = math.fabs(self._p_max - p_t)
            self._window.append(p_t)
            p_min = min(self._window)
            diff = math.fabs(p_min - p_t)
            # print('stabilization diff = ', diff)
            if diff < self._epsilon_s:
                self._batch_counter += 1
            else:
                self._batch_counter = 0

            if self._batch_counter >= self._stabilization_threshold:
                self._stabilization_phase = True
                self._drift_started = False
                self._batch_counter = 0

        if self._p_max < p_t:
            self._p_max = p_t
        drift_started = (self._p_max - p_t) >= self._epsilon

        if drift_started:
            self._drift_started = True
            self._drift_phase = True
            self._p_max = 0
            self._batch_counter = 0

    def change_detected(self):
        return self._drift_phase

    def stabilization_detected(self):
        return self._stabilization_phase

    @property
    def batch_size(self):
        return self._window_size

    @batch_size.setter
    def batch_size(self, value):
        self._window_size = value
        self._epsilon = math.sqrt(math.log((1 / self._delta), math.e) / (2 * self._window_size))
        # self._epsilon_s = self._epsilon
