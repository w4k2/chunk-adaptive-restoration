configs = {
    'stream_learn_recurring_abrupt_1': {
        'chunk_size': 1000,
        'drift_chunk_size': 100,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': True,
        'incremental': False,
        'concept_sigmoid_spacing': None,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 1,
    },
    'stream_learn_recurring_abrupt_2': {
        'chunk_size': 500,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': True,
        'incremental': False,
        'concept_sigmoid_spacing': None,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 1,
    },
    'stream_learn_recurring_abrupt_3': {
        'chunk_size': 100,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': True,
        'incremental': False,
        'concept_sigmoid_spacing': None,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 4,
    },
    'stream_learn_nonrecurring_abrupt_1': {
        'chunk_size': 1000,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': False,
        'concept_sigmoid_spacing': None,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 5,
    },
    'stream_learn_nonrecurring_abrupt_2': {
        'chunk_size': 500,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': False,
        'concept_sigmoid_spacing': None,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 9,
    },
    'stream_learn_nonrecurring_abrupt_3': {
        'chunk_size': 100,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': False,
        'concept_sigmoid_spacing': None,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 10,
    },
    'stream_learn_nonrecurring_gradual_1': {
        'chunk_size': 1000,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': False,
        'concept_sigmoid_spacing': 8.0,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 2,
    },
    'stream_learn_nonrecurring_gradual_2': {
        'chunk_size': 500,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': False,
        'concept_sigmoid_spacing': 5.0,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 14,
    },
    'stream_learn_nonrecurring_gradual_3': {
        'chunk_size': 100,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': False,
        'concept_sigmoid_spacing': 5.0,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 15,
    },
    'stream_learn_nonrecurring_incremental_1': {
        'chunk_size': 1000,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': True,
        'concept_sigmoid_spacing': 5.0,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.0001,
        'random_state': 1,
    },
    'stream_learn_nonrecurring_incremental_2': {
        'chunk_size': 500,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': True,
        'concept_sigmoid_spacing': 5.0,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 17,
    },
    'stream_learn_nonrecurring_incremental_3': {
        'chunk_size': 100,
        'drift_chunk_size': 30,
        'n_chunks': 300,
        'n_drifts': 5,
        'recurring': False,
        'incremental': True,
        'concept_sigmoid_spacing': 5.0,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'random_state': 18,
    },
    'insects_1': {
        'chunk_size': 1000,
        'drift_chunk_size': 30,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'n_drifts': 2,
    },
    'insects_2': {
        'chunk_size': 500,
        'drift_chunk_size': 30,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'n_drifts': 2,
    },
    'insects_3': {
        'chunk_size': 100,
        'drift_chunk_size': 30,
        'fhdsdm_window_size_drift': 1000,
        'fhdsdm_window_size_stabilization': 30,
        'fhdsdm_epsilon_s': 0.001,
        'n_drifts': 2,
    }
}
