import numpy as np

from reservoirpy import datasets

def get_mg_train_test_splits(u0=None, dt=1.0,
                             train_timesteps=1000, test_timesteps=1000,
                             pred_horizon=500, forecasting_method='multi_step'):
    
    '''
    Docstring for get_mg_train_test_splits
    
    :param u0: Description
    :param dt: Description
    :param train_timesteps: Description
    :param test_timesteps: Description
    :param pred_horizon: Description
    :param multi_step_prediction: when 'multi_step' means that from the state t=t_0 we predict the next h={1, 2, ... pred_horizions}.  When 'single_step' means that we only do single step predictions, and feed the prediction back to the model to forecast all the prediction horizon.
    :param device: Description
    '''

    assert forecasting_method in ['multi_step', 'single_step']

    mg = datasets.mackey_glass(
        n_timesteps = 200000,
        tau = 17,
        a = 0.2,
        b = 0.1,
        n = 10,
        x0 = u0, # initial condition
        h = dt, # dt
        seed = 42,
        history = None,
    ).flatten()
    mg = (mg-0.35) / (1.4 - 0.4) # simple 'recenter'/scaling to [0, 1]

    assert mg[17:].max() < 1.2 and mg[17:].min() > -0.2

    lyapunov_max = 0.006
    lyapunov_time = 1 / lyapunov_max
    timevec = np.arange(len(mg), dtype=np.float32) * dt / lyapunov_time
    mg = np.vstack((timevec, mg), dtype=np.float32) # First row: time [lyapunov time], Second row: MG time series

    ndim = 1

    # Split train test
    print(f'MG total time steps: {mg.shape[1]}.')

    assert pred_horizon >= 1, "Prediction horizon (h) must be bigger than 1."

    # Test split
    X_test = mg[:, -test_timesteps-pred_horizon:]
    y_test = np.zeros((test_timesteps, ndim*pred_horizon)) # Each column corresponds to h = 1, 2, ... pred_horizon
    for i in range(test_timesteps):
        y_test[i, :] = np.roll(X_test[1, :].flatten(), shift=-1-i)[:pred_horizon].flatten()

    # Train split
    gap = 1 if forecasting_method == 'single_step' else pred_horizon
    X_train = mg[:, -train_timesteps-gap-test_timesteps-pred_horizon:-test_timesteps-pred_horizon] # train data contains the gap points, but they shouldn't be used to train the regression model (only to iterate the reservoir up to the test data)
    y_train = np.zeros((train_timesteps, ndim*gap)) # Each column corresponds to h = 1, 2, ... pred_horizon

    for i in range(train_timesteps):
        y_train[i, :] = np.roll(X_train[1, :].flatten(), shift=-1-i)[:gap].flatten()

    return mg, X_train, y_train, X_test, y_test


def get_mg_train_test_splits2(data_split_params: dict, return_complete_mg: bool = False):
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
    

    mg = (mg-0.35) / (1.4 - 0.4) # simple 'recenter'/scaling to [0, 1]

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