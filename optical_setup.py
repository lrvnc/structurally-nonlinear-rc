import torch
import numpy as np
import torch.nn.functional as F

from LightPipes import Begin, RandomPhase, MultIntensity, Fresnel, Intensity, cm, mm, um, nm
from vimbPy import VimbaXController, OutstreamManager, TIMESTAMP_FREQUENCY
from pyALP41.consts import DMD_HEIGHT, DMD_WIDTH
from multiprocessing import Process, Queue
from time import monotonic, sleep
from encoding import vectorized_basket
from pyALP41 import ALP41Controller
from utils import corr2, max_singular
from threading import Thread, Event
from typing import Dict, Union
from logger import LogHandler
from collections import deque
from visuals import monitor


# TODO: OpticalSetupSim docstrings
class OpticalSetupSim:
    def __init__(self,
                 res_dim: int = 32, state_nbin: int = 10, inpt_portion: float = 2.0,
                 device: str = 'cpu', seed: int = 7, save_speckles: bool = False,
                 load_TMs: str = None):

        assert device == 'cpu', NotImplementedError("Simulation doesn't work on GPU.")

        self.logger = LogHandler(module_name='Optical Setup Simulation', log_dir='./', log_level='INFO')
        self.load_TMs = load_TMs

        self._base_seed = seed
        self.rng = np.random.default_rng(seed)
        self.device = device

        self.res_dim = res_dim
        self.state_nbin = state_nbin
        self.inpt_portion = inpt_portion # -> inpt_portion = 2 corresponds to 1/1 ratio
        _ = self._dummy_mask(verbose=True)
        
        self.save_speckles = save_speckles
        self.speckles_lin = []
        self.speckles_nonlin = []

    
    def _metadata(self):
        return {
            'res_dim': self.res_dim,
            'state_nbin': self.state_nbin,
            'lin_exposure': self.lin_exposure,
            'nonlin_exposure': self.nonlin_exposure,
        }
    

    def _dummy_mask(self, verbose=False):
        '''
        Docstring for _dummy_mask
        
        :param self: Description
        :param verbose: Description
        '''
        temp_state = np.ones(self.res_dim, dtype=np.float32).reshape(-1, 1) * 0.5
        temp_inpt = np.ones(1, dtype=np.float32).reshape(-1, 1) * 0.5

        vec_temp_state = vectorized_basket(array_inpt=temp_state, nbin=self.state_nbin, min_val=-0.5, max_val=1.5).reshape(-1, 1)
        self.inpt_nbin = int(self.inpt_portion * np.count_nonzero(vec_temp_state))
        vec_temp_inpt = vectorized_basket(array_inpt=temp_inpt, nbin=self.inpt_nbin, min_val=-0.5, max_val=1.5).reshape(-1, 1)
        temp_mask = np.vstack((vec_temp_state, vec_temp_inpt))
        self.mask_dim = temp_mask.size # -> DMD area

        if verbose:
            self.logger.info(f"DMD area corresponds to {self.mask_dim} pixels.")
            self.logger.info(f"State corresponds to {100*vec_temp_state.size/self.mask_dim}% of the DMD area. Input to {100*vec_temp_inpt.size/self.mask_dim}%.")
        
        return temp_mask


    def _init_TM(self):
        '''
        Initialize the Transmission Matrices (TMs) of our system.

        Returns:
            self.TM_cam: np.array[complex64] of dimension (self.cam_area, self.dmd_area) -> DMD to Camera
            self.TM_dmd: np.array[complex64] of dimension (self.dmd_area, self.dmd_area) -> DMD to DMD
        '''
        
        assert isinstance(self.load_TMs, Union[str, None]), ValueError('load_TMs must be the path to the TMs.')

        if self.load_TMs is not None:
            tms = np.load(self.load_TMs)
            self.TM_cam = tms['TM_cam']
            self.TM_dmd = tms['TM_dmd']
        else:
            tm_real = self.rng.standard_normal((self.res_dim, self.mask_dim), dtype=np.float32) # -> cam_dim = res_dim in our simplification
            tm_imag = self.rng.standard_normal((self.res_dim, self.mask_dim), dtype=np.float32)
            self.TM_cam = (tm_real + 1j*tm_imag).astype(np.complex64)
            # Spectral adjust
            self.TM_cam = 0.99 * self.TM_cam / max_singular(self.TM_cam, n_iter=100)
            self.logger.info(f"TM from DMD -> Cam created. Shape: {self.TM_cam.shape}, Size: {self.TM_cam.nbytes / 1e9} GB), Type: {self.TM_cam.dtype}")
            del tm_real, tm_imag

            tm_real = self.rng.standard_normal((self.mask_dim, self.mask_dim), dtype=np.float32) # -> dmd_area = mask_dim in our simplification
            tm_imag = self.rng.standard_normal((self.mask_dim, self.mask_dim), dtype=np.float32)
            self.TM_dmd = (tm_real + 1j*tm_imag).astype(np.complex64)
            # Spectral adjust
            self.TM_dmd = 0.99 * self.TM_dmd / max_singular(self.TM_dmd)
            self.logger.info(f"TM from DMD -> DMD created. Shape: {self.TM_dmd.shape}, Size: {self.TM_dmd.nbytes / 1e9} GB), Type: {self.TM_dmd.dtype}")
            del tm_real, tm_imag

    def _adjust_exposure(self, linear: Union[float, str] = 1.0, nonlinear: Union[float, str] = 1.0):
        '''
        Adjust the max. intensity value the camera can capture.

        Params:
            linear: float. Factor by each the max is reduced in the linear path.
            nonlinear: float. Factor by each the max is reduced in the nonlinear path.
        '''
        self.lin_exposure = linear
        self.nonlin_exposure = nonlinear

        # 1. Linear
        if isinstance(linear, float):
            E_out = self.TM_cam @ self._dummy_mask()
            self.max_intensity_lin = np.max(np.abs(E_out)**2) * linear
        else:
            self.max_intensity_lin = 'auto'

        # 2. Nonlinear
        if isinstance(nonlinear, float):
            E_out = self.TM_cam @ ( (self.TM_dmd @ self._dummy_mask()) * self._dummy_mask())
            self.max_intensity_nonlin = np.max(np.abs(E_out)**2) * nonlinear
        else:
            self.max_intensity_nonlin = 'auto'
        
    
    def _gen_speckle(self, E_out: np.ndarray[np.float32], optical_features: str):
        speckle = np.abs(E_out)**2
        
        max_intensity = self.max_intensity_lin if optical_features == 'linear' else self.max_intensity_nonlin
        max_intensity = speckle.max() if isinstance(max_intensity, str) else max_intensity # auto exposure

        speckle_cam = np.clip(speckle, a_min=0, a_max=max_intensity) / max_intensity # Simulate camera acquisition
        speckle_cam = np.rint(speckle_cam * 255, dtype=np.float32)

        if (np.count_nonzero(speckle_cam == 255) / speckle_cam.size) > 0.01:
            self.logger.warning(f'Speckle is saturating! ({np.count_nonzero(speckle_cam == 255)} / {speckle_cam.size})')
            # raise ValueError('Speckle is saturating!')

        return speckle_cam
    
    def compute_f(self, state: np.ndarray[np.float32], inpt: np.ndarray[np.float32], optical_features: str = 'both') -> np.ndarray[np.float32]:
        '''
        Simulates the optical activation function.

        Params:
            state: np.array[float32] of dimension (self.res_dim, 1).
            inpt: np.array[float32] of dimension (self.input_dim, 1).

        Returns:
            speckle_lin: np.array[float32] of dimension (self.cam_h, self.cam_w)
            speckle_nonlin: np.array[float32] of dimension (self.cam_h, self.cam_w)
        '''
        # 1. Generate DMD mask
        dmd_mask = self.generate_dmd_mask(state=state, inpt=inpt)

        # 2. Propagate
        if optical_features == 'linear':
            E_out = self.TM_cam @ dmd_mask
            speckle_lin = self._gen_speckle(E_out, optical_features='linear')
            speckle_nonlin = None
        elif optical_features == 'nonlinear':
            speckle_lin = None
            E_out = self.TM_cam @ ( (self.TM_dmd @ dmd_mask) * dmd_mask ) # Try on ON other OFF #! E essa TM_cam, n tem esse difusor ai!... deveria usar um propagation beam model? sei la...
            speckle_nonlin = self._gen_speckle(E_out, optical_features='nonlinear')
        elif optical_features == 'both':
            E_out = self.TM_cam @ dmd_mask
            speckle_lin = self._gen_speckle(E_out, optical_features='linear')
            E_out = self.TM_cam @ ( (self.TM_dmd @ dmd_mask) * dmd_mask )
            speckle_nonlin = self._gen_speckle(E_out, optical_features='nonlinear')
        else:
            raise NotImplementedError('What kind of magical features are you looking for?')

        if self.save_speckles:
            self.speckles_lin.append(speckle_lin)
            self.speckles_nonlin.append(speckle_nonlin)
        
        return speckle_lin, speckle_nonlin
    

    def generate_dmd_mask(self, state: np.ndarray[np.float32], inpt: np.ndarray[np.float32]) -> np.ndarray[np.float32]:
        state_bin = vectorized_basket(array_inpt=state, nbin=self.state_nbin, min_val=-0.5, max_val=1.5, clip=False).reshape(-1, 1)
        inpt_bin = vectorized_basket(array_inpt=inpt.reshape(1, -1), nbin=self.inpt_nbin, min_val=-0.5, max_val=1.5, clip=False).reshape(-1, 1)
        mask = np.vstack((state_bin, inpt_bin))
        return mask
    

    def reset_speckle_mem(self):
        self.speckles_lin = []
        self.speckles_nonlin = []


class OpticalSetupSim2:
    def __init__(self,
                 grid_points: int = 512, res_dim: int = 32, state_nbin: int = 8,
                 z_dmd_diff: float = 20*cm, z_diff_cam_lin: float = 10*cm,
                 z_diff_dmd_nonlin: float = 20*cm, z_dmd_cam_nonlin: float = 20*cm,
                 wavelength: float = 632.8*nm, save_speckles: bool = False,
                 device: str = 'cpu', seed: int = 7):

        assert device == 'cpu', NotImplementedError("Simulation doesn't work on GPU.")

        self.logger = LogHandler(module_name='Optical Setup Simulation', log_dir='./', log_level='INFO')
        self._micromirror_pitch = 10.8*um

        self._base_seed = seed
        self.device = device

        self.grid_points = grid_points
        self.grid_size = self._micromirror_pitch * self.grid_points

        self.z_dmd_diff = z_dmd_diff
        self.z_diff_cam_lin = z_diff_cam_lin
        self.z_diff_dmd_nonlin = z_diff_dmd_nonlin
        self.z_dmd_cam_nonlin = z_dmd_cam_nonlin
        self.wavelength = wavelength

        self.res_dim = res_dim
        self.state_nbin = state_nbin
        self.inpt_rep = None
        
        self.stable = True
        self.save_speckles = save_speckles
        self.speckles_lin = []
        self.speckles_nonlin = []

    
    def _metadata(self):
        return {
            '_micromirror_pitch': self._micromirror_pitch,
            '_base_seed': self._base_seed,
            'grid_points': self.grid_points,
            'grid_size': self.grid_size,
            'z_dmd_diff': self.z_dmd_diff,
            'z_diff_cam_lin': self.z_diff_cam_lin,
            'z_diff_dmd_nonlin': self.z_diff_dmd_nonlin,
            'z_dmd_cam_nonlin': self.z_dmd_cam_nonlin,
            'wavelength': self.wavelength,
            'res_dim': self.res_dim,
            'state_nbin': self.state_nbin,
            'inpt_rep': self.inpt_rep,
            'max_i_lin': self.max_intensity_lin,
            'max_i_nonlin': self.max_intensity_nonlin,
        }
    

    def _adjust_exposure(self, linear: Union[float, str, None] = None, nonlinear: Union[float, str, None] = None):
        '''
        Adjust the max. intensity value the camera can capture.

        Params:
            linear: float. Factor by each the max is reduced in the linear path.
            nonlinear: float. Factor by each the max is reduced in the nonlinear path.
        '''
        state = np.ones(self.res_dim).reshape(-1, 1) * 0.5
        inpt = np.ones(1).reshape(-1, 1) * 0.5
        speckle_lin, speckle_nonlin = self.compute_f(state=state, inpt=inpt, simulate_cam=False, downsample=False, optical_features='both')
        
        print(f"Max(I_lin) = {speckle_lin.max()}, Max(I_nonlin) = {speckle_nonlin.max()}")

        if linear is None:
            self.max_intensity_lin = speckle_lin.max()
        elif linear  == 'auto':
            self.max_intensity_lin = 'auto'
        elif isinstance(linear, float):
            self.max_intensity_lin = linear
        else:
            raise ValueError('What kind of exposure magical exposure you want?')
        
        if nonlinear is None:
            self.max_intensity_nonlin = speckle_nonlin.max()
        elif nonlinear  == 'auto':
            self.max_intensity_nonlin = 'auto'
        elif isinstance(nonlinear, float):
            self.max_intensity_nonlin = nonlinear
        else:
            raise ValueError('What kind of exposure magical exposure you want?')

        print(f"I_lin max. set to {self.max_intensity_lin}, I_nonlin max. set to {self.max_intensity_nonlin}")
        
    
    def compute_f(self, state: np.ndarray[np.float32], inpt: np.ndarray[np.float32], simulate_cam: bool = True, downsample: bool = False,
                  optical_features: str = 'both') -> np.ndarray[np.float32]:
        '''
        Simulates the optical activation function.

        Params:
            state: np.array[float32] of dimension (self.res_dim, 1).
            inpt: np.array[float32] of dimension (self.input_dim, 1).

        Returns:
            speckle_lin: np.array[float32] of dimension (self.cam_h, self.cam_w)
            speckle_nonlin: np.array[float32] of dimension (self.cam_h, self.cam_w)
        '''
        # 1. Generate DMD mask
        dmd_mask = self.generate_dmd_mask(state=state, inpt=inpt)

        # 2. Propagate
        beam = Begin(size=self.grid_size, labda=self.wavelength, N=self.grid_points, dtype=np.complex64)
        beam = MultIntensity(Fin=beam, Intens=dmd_mask)
        beam = Fresnel(Fin=beam, z=self.z_dmd_diff, usepyFFTW=True) # DMD hit 1, same path lin/nonlin
        beam = RandomPhase(Fin=beam, seed=self._base_seed, maxPhase=2*np.pi) # TODO: we can accelerate by sampling the phases before hand (fixed)

        # Linear path
        if optical_features == 'linear' or optical_features == 'both':
            beam_lin = Fresnel(Fin=beam, z=self.z_diff_cam_lin, usepyFFTW=True)
            speckle_lin = Intensity(Fin=beam_lin, flag=0) # Not normalized
            if simulate_cam:
                speckle_lin = np.clip(a=speckle_lin, a_min=0, a_max=self.max_intensity_lin) / self.max_intensity_lin # Simulating camera saturation
                speckle_lin = np.rint(speckle_lin * 255) # Simulating camera acquisition
                if np.count_nonzero(speckle_lin == 255) / speckle_lin.size > 0.01:
                    raise ValueError("Speckle (linear) is saturating.")
            if self.save_speckles:
                    self.speckles_lin.append(speckle_lin)
            if downsample:
                speckle_lin, _ = circular_downsample(img=speckle_lin, s=16, radius=4, agg='mean')
        else:
            speckle_lin = None

        # Nonlinear path
        if optical_features == 'nonlinear' or optical_features == 'both':
            beam_nonlin = Fresnel(Fin=beam, z=self.z_diff_dmd_nonlin, usepyFFTW=True)
            beam_nonlin = MultIntensity(Fin=beam_nonlin, Intens=dmd_mask)
            beam_nonlin = Fresnel(Fin=beam_nonlin, z=self.z_dmd_cam_nonlin, usepyFFTW=True)
            speckle_nonlin = Intensity(Fin=beam_nonlin, flag=0) # Not normalized
            if simulate_cam:
                speckle_nonlin = np.clip(a=speckle_nonlin, a_min=0, a_max=self.max_intensity_nonlin) / self.max_intensity_nonlin # Simulating camera saturation
                speckle_nonlin = np.rint(speckle_nonlin * 255) # Simulating camera acquisition
                if np.count_nonzero(speckle_nonlin == 255) / speckle_nonlin.size > 0.01:
                    raise ValueError("Speckle (nonlinear) is saturating.")
            if self.save_speckles:
                    self.speckles_nonlin.append(speckle_nonlin)
            if downsample:
                speckle_nonlin, _ = circular_downsample(img=speckle_nonlin, s=16, radius=4, agg='mean')
        else:
            speckle_nonlin = None
        
        return speckle_lin, speckle_nonlin
    

    def generate_dmd_mask(self, state: np.ndarray[np.float32], inpt: np.ndarray[np.float32]) -> np.ndarray[np.float32]:

        # 1. Background
        n_squares = 8
        rows = np.floor_divide(np.arange(self.grid_points), self.grid_points // n_squares, dtype=np.float32).reshape(-1, 1)
        cols = np.floor_divide(np.arange(self.grid_points), self.grid_points // n_squares, dtype=np.float32).reshape(1, -1)
        dmd_mask = (rows + cols) % 2

        # 2. Binarization (basket encoding)
        state_bin = vectorized_basket(array_inpt=state, nbin=self.state_nbin, min_val=-0.5, max_val=1.5, clip=False)
        state_bin = state_bin.reshape(2**5, -1) # -> controls state macropixel size
        state_bin = np.repeat(state_bin, axis=1, repeats=self.grid_points // state_bin.shape[1])

        inpt_bin = vectorized_basket(array_inpt=inpt.reshape(1, -1), nbin=self.grid_points, min_val=-0.5, max_val=1.5, clip=False)
        if self.inpt_rep is None:
            self.inpt_rep = np.count_nonzero(state_bin) // np.count_nonzero(inpt_bin) // 2 # -> change state/input ratio
        inpt_bin = np.repeat(inpt_bin, repeats=self.inpt_rep, axis=0).reshape(-1, inpt.shape[0], self.grid_points).transpose(1, 0, 2).reshape(-1, self.grid_points)

        # 3. Putting everything together
        inner_mask = np.vstack((state_bin, inpt_bin))
        inner_mask = np.repeat(inner_mask, axis=0, repeats=self.grid_points//inner_mask.shape[0])

        dmd_mask[:inner_mask.shape[0], :] = inner_mask
        dmd_mask = np.roll(dmd_mask, (dmd_mask.shape[0] - inner_mask.shape[0]) // 2, axis=0)

        return dmd_mask
    

    def _refresh_ref_speckle(self):
        ...

    
    def _check_stability(self, optical_path: str):
        ...


    def reset_speckle_mem(self):
        self.speckles_lin = []
        self.speckles_nonlin = []


class OpticalSetup:

    def __init__(self,
                 monitoring: bool = True, save_speckles: bool = False,
                 grid_points: int = 1024, state_nbin: int = 16, res_dim: int = 512):

        self.logger = LogHandler(module_name='Optical Setup', log_dir='./', log_level='INFO')
        self._micromirror_pitch = 10.8*um
        self.monitoring = monitoring
        
        self.grid_points = grid_points # total nb. of dmd pixels
        self._grid_size = self._micromirror_pitch * self.grid_points # physical dmd surface used
        
        self.ref_pattern = self.get_mask('checkerboard').copy()
        self.res_dim = res_dim
        self.state_nbin = state_nbin
        self.inpt_rep = None

        self.check_stability_counter = 0
        self.save_speckles = save_speckles
        self.speckles_lin, self.corr_lin = [], []
        self.speckles_nonlin, self.corr_nonlin = [], []


    def _metadata(self):
        return {
            '_micromirror_pitch': self._micromirror_pitch,
            'grid_points': self.grid_points,
            'grid_size': self._grid_size,
            'z_dmd_diff': None,
            'z_diff_cam_lin': None,
            'z_diff_dmd_nonlin': None,
            'z_dmd_cam_nonlin': None,
            'wavelength': 632.8*nm,
            'res_dim': self.res_dim,
            'state_nbin': self.state_nbin,
            'inpt_rep': self.inpt_rep,
            'max_i_lin': 'See VimbaX',
            'max_i_nonlin': 'See VimbaX',
        }


    def _on(self):
        self._dmd_on()
        self._cams_on()
        if self.monitoring:
            self._monitor_on()

    
    def _off(self):
        self._dmd_off()
        self._cams_off()
        if self.monitoring:
            self._monitor_off()


    def _cams_on(self):
        self.cams: Dict[str, Dict[str, Union[str, Event, OutstreamManager, Thread]]] = {
            'single': {'cam_id': 'DEV_000F314E71C8', 'stop': Event(), 'ready': Event(), 'out_manager': OutstreamManager()},
            'double': {'cam_id': 'DEV_000F314DE426', 'stop': Event(), 'ready': Event(), 'out_manager': OutstreamManager()},
            }
        
        for cam_name, cam in self.cams.items():
            cam['stop'].clear()
            cam['ready'].clear()
            cam['out_manager'].create(f'cam_out', maxsize=1, start_enabled=True)

            self.cams_thread = Thread(target=self.__cam_task, daemon=True)
        
        self.cams_thread.start()

        self.q_single, _ = self.cams['single']['out_manager'].get('cam_out')
        self.q_double, _ = self.cams['double']['out_manager'].get('cam_out')
        
        while not self.cams['single']['ready'].is_set():
            self.cams['single']['ready'].wait(.5)
        while not self.cams['double']['ready'].is_set():
            self.cams['double']['ready'].wait(.5)


    def __cam_task(self):
        with VimbaXController(cam_id=self.cams["single"]["cam_id"], log_dir="./", log_level="INFO") as ctrl_single,\
             VimbaXController(cam_id=self.cams["double"]["cam_id"], log_dir="./", log_level="INFO") as ctrl_double:

            ctrl_controllers = {
                "single": ctrl_single,
                "double": ctrl_double,
            }

            threads = []

            for cam_name, cam_info in self.cams.items():
                ctrl = ctrl_controllers[cam_name]

                def record_wrapper(ctrl_ref, cam_info_ref):
                    with ctrl_ref.cam as cam1:
                        cam1.TriggerSource.set('Line1')
                        cam1.TriggerMode.set('On')
                    
                    ctrl_ref.record(
                        stop_event=cam_info_ref['stop'],
                        ready_event=cam_info_ref['ready'],
                        outstream_manager=cam_info_ref['out_manager'],
                    )

                t = Thread(target=record_wrapper, args=(ctrl, cam_info), daemon=True)
                t.start()
                cam_info["thread"] = t
                threads.append(t)

            for t in threads:
                t.join()

    
    def _cams_off(self):
        buffer = []
        for c in self.cams.keys():
            q, _ = self.cams[c]['out_manager'].get('cam_out')

            while not q.empty():
                buffer.append(q.get())

            if len(buffer) > 0:
                print(f'Imgs on the queue before turning off: {len(buffer)} - {c}')

        for name, cam in self.cams.items():
            cam['stop'].set()
            cam['thread'].join(timeout=2.0)
        self.cams_thread.join(timeout=2.0)

    
    def _adjust_exposure(self, linear: Union[float, str, None] = None, nonlinear: Union[float, str, None] = None):
        ...


    def _dmd_on(self):
        self.dmd = ALP41Controller(log_dir="./", log_level="INFO").connect(device_sn=10071)
        sbuffer = []
        for _ in range(5):
            sbuffer.append(self.dmd.alloc_memory(bit_planes=1, pic_num=1, bin_mode=True))
        self.sbuffer = np.array(sbuffer).flatten()

    
    def _dmd_off(self):
        self.dmd.idle()
        for s in self.sbuffer:
            self.dmd.free_memory(int(s))
        self.dmd.disconnect()

    
    def _dmd_display(self, pattern: np.ndarray, continuous: bool = True):
        self.seq_id = int(self.sbuffer[0])
        self.dmd.send_imgs_to_mem(seq_id=self.seq_id, img_array=pattern, pic_offset=0)
        self.dmd.display(seq_id=self.seq_id, continuous=continuous)
        self.sbuffer = np.roll(self.sbuffer, 1)

    
    def _dmd_warmup(self, horizon: int = 20, criterion: float = 0.99, period: int = 60):
        imgs_single, imgs_double = deque(maxlen=horizon), deque(maxlen=horizon)
        ts_single, ts_double = deque(maxlen=horizon), deque(maxlen=horizon)

        next_deadline = monotonic() + period

        while True:
            # Task
            self._dmd_display(pattern=self.ref_pattern[np.newaxis, np.newaxis, :, :], continuous=False)

            ts_s, img_single = self.q_single.get()
            ts_d, img_double = self.q_double.get()

            imgs_single.append(img_single.squeeze())
            ts_single.append(ts_s)

            imgs_double.append(img_double.squeeze())
            ts_double.append(ts_d)

            stability_corr_buffer_single = []
            for speckle in imgs_single:
                stability_corr_buffer_single.append(corr2(imgs_single[0], speckle))

            stability_corr_buffer_double = []
            for speckle in imgs_double:
                stability_corr_buffer_double.append(corr2(imgs_double[0], speckle))
            
            if len(stability_corr_buffer_single) == horizon:
                stability_condition = all([i > criterion for i in stability_corr_buffer_single]) and all([i > criterion for i in stability_corr_buffer_double])
                if stability_condition:
                    self.logger.info(f'DMD is ready.')
                    break
            
            if self.monitoring:
                self.mask_q.put(self.ref_pattern)
                self.spkl_q.put(img_single.squeeze().astype(np.uint8)) # Single
                self.spkl2_q.put(img_double.squeeze().astype(np.uint8)) # Double
                self.corr_q.put((ts_single[-1]/(TIMESTAMP_FREQUENCY*60), stability_corr_buffer_single[-1], ts_double[-1]/(TIMESTAMP_FREQUENCY*60), stability_corr_buffer_double[-1]))

            # Scheduler
            next_deadline += period

            sleep_time = next_deadline - monotonic()
            if sleep_time > 0:
                sleep(sleep_time)
            else:
                missed = int(-sleep_time // period) + 1
                next_deadline += missed * period
        
        self._refresh_ref_speckle() # Update ref. speckle patterns

    
    def _monitor_on(self):
        self.mask_q, self.corr_q, self.spkl_q, self.spkl2_q, self.stts_q = Queue(), Queue(), Queue(), Queue(), Queue()
        self.monitor_thread = Process(target=monitor, args=(self.mask_q, self.corr_q, self.spkl_q, self.spkl2_q, self.stts_q), daemon=True)
        self.monitor_thread.start()


    def _monitor_off(self):
        self.monitor_thread.terminate()
        self.monitor_thread.join(timeout=0.1)


    def _test_monitor(self):
        for i in range(500):
            self.mask_q.put(np.random.uniform(0, 255, (1080, 1920)).astype(np.uint8))
            self.spkl_q.put(np.clip(np.random.normal(loc=130, scale=50, size=(300, 300)).astype(np.uint8)), 0, 255)
            self.spkl2_q.put(np.clip(np.random.normal(loc=130, scale=50, size=(300, 300)).astype(np.uint8)), 0, 255)
            self.corr_q.put((i, 0.75, i, 0.25))
            self.stts_q.put((i, np.random.random(5)))
            sleep(0.1)


    def generate_dmd_mask(self, state: np.ndarray[np.float32], inpt: np.ndarray[np.float32], verbose: bool = False) -> np.ndarray[np.float32]:
        
        # 1. Background (bias)
        dmd_mask = self.ref_pattern.copy().T

        # 2. Binarization (basket encoding)
        # state_dummy = np.ones_like(state) * 0.5
        state_bin = vectorized_basket(array_inpt=state, nbin=self.state_nbin, min_val=-0.5, max_val=1.5, clip=False)
        state_bin = state_bin.reshape(2**5, -1) # -> controls state macropixel size
        state_bin = np.repeat(state_bin, axis=1, repeats=self.grid_points // state_bin.shape[1])

        inpt_bin = vectorized_basket(array_inpt=inpt.reshape(1, -1), nbin=self.grid_points, min_val=-0.5, max_val=1.5, clip=False)
        if self.inpt_rep is None:
            self.inpt_rep = np.count_nonzero(state_bin) // np.count_nonzero(inpt_bin) # -> change state/input ratio
        inpt_bin = np.repeat(inpt_bin, repeats=self.inpt_rep, axis=0).reshape(-1, inpt.shape[0], self.grid_points).transpose(1, 0, 2).reshape(-1, self.grid_points)

        # 3. Putting everything together
        inner_mask = np.vstack((state_bin, inpt_bin))
        inner_mask = np.repeat(inner_mask, axis=0, repeats=self.grid_points//inner_mask.shape[0]) * 255

        # 5. Add bias background and center
        dmd_mask[:inner_mask.shape[0], (DMD_HEIGHT-self.grid_points)//2:(DMD_HEIGHT-self.grid_points)//2+self.grid_points] = inner_mask
        dmd_mask = np.roll(dmd_mask, (dmd_mask.shape[0] - inner_mask.shape[0]) // 2, axis=0).T
        dmd_mask = dmd_mask[np.newaxis, np.newaxis, :, :]

        if verbose:
            self.logger.info(f"Active mirrors in the state vs. input: {100*np.count_nonzero(state_bin) / np.count_nonzero(inpt_bin):.2f}%")

        return dmd_mask.astype(np.uint8)


    def get_mask(self, id: str = None) -> np.ndarray:
        if id == 'checkerboard':
            n_squares = 5
            rows = np.floor_divide(np.arange(DMD_HEIGHT), DMD_HEIGHT // n_squares).reshape(-1, 1)  # shape (H,1)
            cols = np.floor_divide(np.arange(DMD_WIDTH), DMD_WIDTH // n_squares).reshape(1, -1)    # shape (1,W)
            dmd_mask = (rows + cols) % 2
            dmd_mask = (dmd_mask*255).astype(np.uint8)
        return dmd_mask


    def compute_f(self, state: np.ndarray[np.float32], inpt: np.ndarray[np.float32],
                  optical_features: str = 'linear') -> np.ndarray:
        '''
        Optical activation.

        Params:
            state: np.array[float32] of dimension (self.res_dim, 1).
            inpt: np.array[float32] of dimension (self.input_dim, 1).

        Returns:
            speckle: np.array[float32] of dimension (self.cam_h, self.cam_w)
        '''

        assert optical_features in ['linear', 'nonlinear', 'extended'], 'What kind of magical features are you looking for?'
        
        # 1. Generate DMD mask
        dmd_mask = self.generate_dmd_mask(state=state, inpt=inpt)

        # 2. Propagate
        self._dmd_display(pattern=dmd_mask, continuous=False)
        _, speckle_single = self.q_single.get()
        _, speckle_double = self.q_double.get()

        speckle_single = speckle_single.squeeze().astype(np.float32)
        speckle_double = speckle_double.squeeze().astype(np.float32)

        if self.monitoring:
            self.mask_q.put(dmd_mask.squeeze())
            self.spkl_q.put(speckle_single) # Single
            self.spkl2_q.put(speckle_double) # Double

        if self.check_stability_counter == 10:
            self._check_stability(optical_path=optical_features)
            self.check_stability_counter = 0
        else:
            self.check_stability_counter += 1

        if self.save_speckles:
            self.speckles_lin.append(speckle_single)
            self.speckles_nonlin.append(speckle_double)
        
        return speckle_single, speckle_double
    

    def _refresh_ref_speckle(self):
        self._dmd_display(pattern=self.ref_pattern[np.newaxis, np.newaxis, :, :], continuous=False)

        ts_s, img_single = self.q_single.get()
        ts_d, img_double = self.q_double.get()

        self.ref_speckle_single, self.ref_speckle_double = img_single.squeeze(), img_double.squeeze()
        self.ref_speckle_timestamp_single, self.ref_speckle_timestamp_double = ts_s, ts_d

    
    def _check_stability(self, optical_path: str, raise_error_on: Union[float, None] = 0.95):

        assert optical_path in ['linear', 'nonlinear'], 'What kind of magical features are you looking for?'

        self._dmd_display(pattern=self.ref_pattern[np.newaxis, np.newaxis, :, :], continuous=False)

        ts_s, img_single = self.q_single.get()
        ts_d, img_double = self.q_double.get()

        single_corr = corr2(img1=self.ref_speckle_single, img2=img_single.squeeze())
        double_corr = corr2(img1=self.ref_speckle_double, img2=img_double.squeeze())

        if self.monitoring:
            self.corr_q.put((ts_s/(TIMESTAMP_FREQUENCY*60), single_corr, ts_d/(TIMESTAMP_FREQUENCY*60), double_corr))

        self.corr_lin.append(single_corr)
        self.corr_nonlin.append(double_corr)

        if raise_error_on is not None:
            corr = single_corr if optical_path == 'linear' else double_corr
            if corr < raise_error_on:
                raise ValueError(f'Correlation dropped below {raise_error_on}. Correlation: {corr}')


    def reset_speckle_mem(self):
        self.speckles_lin, self.corr_lin = [], []
        self.speckles_nonlin, self.corr_nonlin = [], []


    def _save(self):
        pass
