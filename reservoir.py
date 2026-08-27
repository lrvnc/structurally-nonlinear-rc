import numpy as np

from sklearn.linear_model import Ridge, RidgeCV, LassoCV, ElasticNetCV, BayesianRidge, LinearRegression
from sklearn.neural_network import MLPRegressor
from optical_setup import OpticalSetup
from typing import Union, Callable
from utils import grid_downsample
from tqdm import tqdm


class Reservoir:
    functions = {
        'sin': np.sin, # [-1, 1]
        'sin2': lambda x: np.sin(np.pi/2 * x)**2,
        'cos': np.cos, # [-1, 1]
        'tanh': np.tanh, # [-1, 1]
        'abs': np.abs, # [0, inf]
        'identity': lambda x: x,
        'norm255': lambda x: x/255,
        'sqrt': lambda x: np.sqrt(x/255),
        'trt': lambda x: (x/255)**(1/3),
        'qrt': lambda x: (x/255)**(1/4),
        'wtf': lambda x: np.tanh(5*np.sqrt(x/255)-0.5*5)*0.5 + 0.5
    }

    reg_models = {
        'ridge': Ridge,
        'ridgeCV': RidgeCV, # Pick best alpha automatically
        'lassoCV': LassoCV,
        'elasticNetCV': ElasticNetCV,
        'bayesianRidge': BayesianRidge,
        'ordinary': LinearRegression,
        'mlp': MLPRegressor,
    }
    
    # To test: res_scale; concat u with states; smooth states; train regression only with a window; retrain outlayer
    def __init__(self, res_dim: int, input_dim: int,
                 data_split_params: dict, forecasting_method: str,
                 leaky_rate: int, activation_func: Union[str, Callable], encoding_func: Union[str, Callable],
                 reg_model: str, reg_model_params: dict = None,
                 res_type: str = 'vanilla',
                 optical_features: str = 'linear',
                 optical_setup: Union[None, OpticalSetup] = None,
                 seed: int = 42):
        '''
        Params:
            res_dim: int. Reservoir dimension.
            input_dim: int. Input dimension.
            leaky_rate: float. Modulates the changing rate of the reservoir states.
            encoding_func: str. Encoding function toat operates on the input and state. Default: identity.
            activation_func: str. Activation function of the reservoir. Default: normalize by 255.
            reg_model: int. Which regression model to use as the output layer.
            reg_model_params: dict. Dictionary containing all parameters of the chosen regression model.
        '''

        assert optical_features in ['linear', 'nonlinear', None], NotImplementedError('What kind of magical features are you looking for?')

        self._base_seed = seed
        self.rng = np.random.default_rng(seed)

        self.res_dim = res_dim
        self.input_dim = input_dim

        self.data_split_params = data_split_params
        self.forecasting_method = forecasting_method
        self.forget = self.data_split_params['train_phase']['forget']

        self.leaky_rate = leaky_rate
        self.activation_func = activation_func if callable(activation_func) else Reservoir.functions.get(activation_func, None)
        self.encoding_func = encoding_func if callable(encoding_func) else Reservoir.functions.get(encoding_func, None)
        
        self.reg_model_params = reg_model_params
        self.reg_model = Reservoir.reg_models.get(reg_model, None)
        if self.reg_model is not None:
            if reg_model_params is not None:
                self.reg_model = self.reg_model(**reg_model_params)
            else:
                self.reg_model = self.reg_model()
        
        self.res_type = res_type
        self.optical_setup = optical_setup
        self.optical_features = optical_features
        
        self.optical_setup._refresh_ref_speckle()
        self.speckle_grain_radius = 3
        speckle_tmp = grid_downsample(img=self.optical_setup.ref_speckle_single, s=2*2*self.speckle_grain_radius, radius=self.speckle_grain_radius, agg='mean') # s=8, r=2
        self.speckle_idx = np.random.choice(speckle_tmp.size, size=self.res_dim)
        self.downsample = lambda img: grid_downsample(img=img, s=2*2*self.speckle_grain_radius, radius=self.speckle_grain_radius, agg='mean').flatten()[self.speckle_idx]

    
    def res_dynamics(self, state: np.ndarray[np.float64],
                     speckle_lin: np.ndarray[np.float64] = None, speckle_nonlin: np.ndarray[np.float64] = None) -> np.ndarray[np.float64]:
        '''
        Dynamics of the reservoir (how the states evolve).

        Params:
            state: np.array[float64] of dimension (self.res_dim, 1).
            inpt: np.array[float64] of dimension (self.input_dim, 1).

        Returns:
            next_state: np.array[float64] of dimension(1, self.res_dim)
        '''

        if self.optical_features == 'linear': # Optical implementation
            lin_optical_features = self.downsample(img=speckle_lin)
            next_state = (1-self.leaky_rate) * state + self.leaky_rate * self.activation_func(lin_optical_features).reshape(-1, 1)
        elif self.optical_features == 'nonlinear':
            nonlin_optical_features = self.downsample(img=speckle_nonlin)
            next_state = (1-self.leaky_rate) * state + self.leaky_rate * self.activation_func(nonlin_optical_features).reshape(-1, 1)
        else:
            raise NotImplementedError()
        return next_state
    

    def train_output_layer(self, states: np.ndarray, targets: np.ndarray):
        '''
        Train output layer.

        Params:
            states: tensor[float64] of dimension (T, self.res_dim + extras).
            targets: tensor[float64] of dimension (T, self.pred_horizon).
        '''
        
        # H-steps ahead
        self.reg_model.fit(states, targets)
        self.W_out_multi = self.reg_model.coef_.T.copy() # (N_neurons, h)
        self.b_out_multi = self.reg_model.intercept_.copy()

        # One step ahead (being purist to not make any data leakage)
        self.reg_model.fit(states, targets[:, :self.input_dim])
        self.W_out_one = self.reg_model.coef_.T.copy() # (N_neurons, input_dim)
        self.b_out_one = self.reg_model.intercept_.copy()
    

    def fit(self, train_data: np.ndarray[np.float64], targets: np.ndarray[np.float64]):
        '''
        Iterate the reservoir dynamics and train the output layer.

        Params:
            train_data: np.array[float64] of dimension (self.input_dim, T)
            targets: np.array[float64] of dimension 
            reg_model: str. Which regression model to use as the output layer.
        '''

        assert train_data.ndim == 2 and train_data.shape[0] == self.input_dim, 'Train data must match dimension (self.input_dim, T)'
        
        train_timesteps = self.data_split_params['train_phase']['number_train_timesteps']
        h_max = self.data_split_params['test_phase']['prediction_horizon']

        x = self.rng.uniform(size=(self.res_dim, 1)).astype(np.float64) # Initial state => x(-1)
        self.init_train_state = x.copy()
        self.train_states = np.zeros((train_timesteps+h_max, self.res_dim), dtype=np.float64) # States in rows (top to bottom = x(0) -> x(T))

        for t in tqdm(range(train_timesteps+h_max), desc='Train phase'):
            if self.res_type == 'optical':
                # Need to send the state and the input as colum vectors
                speckle_lin, speckle_nonlin = self.optical_setup.compute_f(state=self.encoding_func(x.reshape(-1, 1)),
                                                                           inpt=self.encoding_func(train_data[:, t].reshape(-1, 1)),
                                                                           optical_features=self.optical_features)
                x = self.res_dynamics(state=x.reshape(-1, 1), speckle_lin=speckle_lin, speckle_nonlin=speckle_nonlin)
            else:
                raise NotImplementedError

            self.train_states[t, :] = x.flatten()

        self.last_train_state = x.copy()

        self.train_output_layer(states=self.train_states[self.forget:-h_max, :], targets=targets[self.forget:, :])


    def predict(self, warmup_data: np.ndarray[np.float64], test_data: np.ndarray[np.float64], random_init: bool = False) -> np.ndarray[np.float64]:
        
        assert test_data.ndim == 2 and test_data.shape[0] == self.input_dim, 'Test data must match dimension (self.input_dim, T)'
        assert warmup_data.ndim == 2 and warmup_data.shape[0] == self.input_dim, 'Test data must match dimension (self.input_dim, T)'

        x = self.last_train_state if not random_init else self.rng.uniform(size=(self.res_dim, 1)).astype(np.float64)
        
        warmup_timesteps = self.data_split_params['test_phase']['number_warmup_timesteps']
        prediction_horizon = self.data_split_params['test_phase']['prediction_horizon']
        N_o = self.data_split_params['test_phase']['number_forecast_origins']
        s = self.data_split_params['test_phase']['forecast_origin_spacing']
        rolling_window = N_o*(1+s)-s

        self.test_warmup_states = np.zeros((warmup_timesteps, self.res_dim), dtype=np.float64) # States in rows
        self.test_states = np.zeros((rolling_window, self.res_dim), dtype=np.float64) # States in rows

        # Warmup (phase 1)
        for t in tqdm(range(warmup_timesteps), desc='Test phase (warm up)'):

            if self.res_type == 'optical':
                # Need to send the state and the input as colum vectors
                speckle_lin, speckle_nonlin = self.optical_setup.compute_f(state=self.encoding_func(x.reshape(-1, 1)),
                                                                           inpt=self.encoding_func(warmup_data[:, t].reshape(-1, 1)),
                                                                           optical_features=self.optical_features)
                x = self.res_dynamics(state=x.reshape(-1, 1), speckle_lin=speckle_lin, speckle_nonlin=speckle_nonlin)
            else:
                raise NotImplementedError
            
            self.test_warmup_states[t, :] = x.flatten()

        # Forecasting (phase 2)
        for t in tqdm(range(rolling_window), desc='Test phase (forecasting)'):

            if self.res_type == 'optical':
                # Need to send the state and the input as colum vectors
                speckle_lin, speckle_nonlin = self.optical_setup.compute_f(state=self.encoding_func(x.reshape(-1, 1)),
                                                                           inpt=self.encoding_func(test_data[:, t].reshape(-1, 1)),
                                                                           optical_features=self.optical_features)
                x = self.res_dynamics(state=x.reshape(-1, 1), speckle_lin=speckle_lin, speckle_nonlin=speckle_nonlin)
            else:
                raise NotImplementedError
            
            self.test_states[t, :] = x.flatten()

        # Predictions
        if 'multi_step' in self.forecasting_method: # direct, fast
            self.predictions_multi = self.test_states[::s+1, :] @ self.W_out_multi + self.b_out_multi
            
        if 'one_step' in self.forecasting_method: # iterative, closed-loop, slow
            self.predictions_one = np.zeros((N_o, prediction_horizon*self.input_dim), dtype=np.float64)
            
            for t_idx, t in enumerate(range(0, rolling_window, s+1)):
                x = self.test_states[t,:].copy()
                for i in tqdm(range(0, prediction_horizon*self.input_dim, self.input_dim), desc=f'Forecasting ({t_idx+1}/{N_o})'):
                    pred = x.flatten() @ self.W_out_one + self.b_out_one
                    self.predictions_one[t_idx, i:i+self.input_dim] = pred.flatten()

                    if self.res_type == 'optical':
                        # Need to send the state and the input as colum vectors
                        pred = pred.reshape(-1, 1)
                        pred = np.clip(pred, a_min=0.0, a_max=1.0)
                        speckle_lin, speckle_nonlin = self.optical_setup.compute_f(state=self.encoding_func(x.reshape(-1, 1)),
                                                                                   inpt=self.encoding_func(pred),
                                                                                   optical_features=self.optical_features)
                        x = self.res_dynamics(state=x.reshape(-1, 1), speckle_lin=speckle_lin, speckle_nonlin=speckle_nonlin)

                    else:
                        raise NotImplementedError    