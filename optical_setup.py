import numpy as np

from multiprocessing import Process, Queue
from encoding import vectorized_basket
from threading import Thread, Event
from time import monotonic, sleep
from typing import Dict, Union
from collections import deque
from visuals import monitor
from utils import corr2

# Internal lab library to control the cameras
from vimbPy import VimbaXController, OutstreamManager, TIMESTAMP_FREQUENCY

# Internal lab library to control the DMD
from pyALP41.consts import DMD_HEIGHT, DMD_WIDTH
from pyALP41 import ALP41Controller 


class OpticalSetup:

    def __init__(self,
                 monitoring: bool = True, save_speckles: bool = False,
                 grid_points: int = 1024, state_nbin: int = 16, res_dim: int = 512):

        self._micromirror_pitch = 10.8 * 1e-6
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
            'wavelength': 632.8 * 1e-9,
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
                    print(f'DMD is ready.')
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
            print(f"Active mirrors in the state vs. input: {100*np.count_nonzero(state_bin) / np.count_nonzero(inpt_bin):.2f}%")

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
