import numpy as np

from reservoirpy import datasets


def get_mg_train_test_splits(data_split_params: dict, return_complete_mg: bool = False):
    # dt = 1 -> 0.006 lyapunov times per time step => 1000 time steps -> ~6 lyapunov times
    # Completely independent train and test splits, no data leakage.

    N_train = data_split_params['train_phase']['number_train_timesteps']
    gap = data_split_params['train_phase']['gap']

    h_max = data_split_params['test_phase']['prediction_horizon']
    N_test_warmup = data_split_params['test_phase']['number_warmup_timesteps']
    N_o = data_split_params['test_phase']['number_forecast_origins']
    s = data_split_params['test_phase']['forecast_origin_spacing']
    rolling_window = N_o*(1+s)-s
    N_test = N_test_warmup + rolling_window + h_max

    stationarity_condition_timesteps = 100000 # corresponding to ~600 lyapunov times

    mg = datasets.mackey_glass(
            n_timesteps = stationarity_condition_timesteps+N_test,
            tau = 17,
            a = 0.2,
            b = 0.1,
            n = 10,
            x0 = data_split_params['train_phase']['x0'], # initial condition
            h = 1.0, # dt
            seed = data_split_params['train_phase']['seed'],
            history = None,
        ).flatten()
    

    mg = mg-0.35 # simple 'recenter'/scaling to [0, 1]

    assert mg[stationarity_condition_timesteps:].max() < 1.2 and mg[stationarity_condition_timesteps:].min() > -0.2

    mg = np.vstack((np.arange(len(mg)), mg))

    # Test split
    X_test_warmup = mg[:, -N_test:-(h_max+rolling_window)]
    X_test = mg[:, -(h_max+rolling_window):]

    y_test = np.zeros((rolling_window, h_max))
    for i in range(rolling_window):
        y_test[i, :] = np.roll(X_test[1, :].flatten(), shift=-1-i)[:h_max].flatten()

    X_test = X_test[:, :rolling_window]

    # Train split
    X_train = mg[:, -N_test-gap-h_max-N_train:-N_test-gap]
    y_train = np.zeros((N_train, h_max)) # Each column corresponds to h = 1, 2, ... pred_horizon

    for i in range(N_train):
        y_train[i, :] = np.roll(X_train[1, :].flatten(), shift=-1-i)[:h_max].flatten()
    
    if return_complete_mg:
        return mg, X_train, y_train, X_test_warmup, X_test, y_test

    return X_train, y_train, X_test_warmup, X_test, y_test