from multiprocessing import Process, Queue
from collections import deque

import numpy as np
import time


def monitor(mask_queue: Queue, corr_queue: Queue, speckle_queue: Queue, speckle2_queue: Queue, states_queue: Queue):
    from vispy import app, scene
    app.use_app('pyqt6')

    # DMD mask
    mask_canvas = scene.SceneCanvas(keys='interactive', show=True, title='Mask monitor', bgcolor='black', size=(800, 400))
    mask_grid = mask_canvas.central_widget.add_grid(margin=0)
    mask_grid.spacing = 0

    title = scene.Label("DMD Mask", color='white', font_size=12)
    title.height_max = 60
    mask_grid.add_widget(title, row=0, col=0)

    mask_view = mask_grid.add_view(row=1, col=0)
    mask_view.camera = 'panzoom'

    temp = np.zeros((1080, 1920), dtype=np.uint8)
    temp[:, :1920//2] = 255
    mask_img = scene.visuals.Image(temp, parent=mask_view.scene, cmap='viridis')
    mask_view.camera.set_range(margin=0)
    mask_view.camera.aspect = 1920 / 1080

    # Speckle
    speckle_canvas = scene.SceneCanvas(keys='interactive', show=True, title='Speckle monitor', bgcolor='black', size=(800, 400))
    speckle_grid = speckle_canvas.central_widget.add_grid(margin=0)
    speckle_grid.spacing = 0

    title1 = scene.Label("Single Pass", color='white', font_size=12); title1.height_max = 60
    title2 = scene.Label("Double Pass", color='white', font_size=12); title2.height_max = 60
    speckle_grid.add_widget(title1, row=0, col=0)
    speckle_grid.add_widget(title2, row=0, col=2)

    speckle_view1 = speckle_grid.add_view(row=1, col=0)
    speckle_view1.camera = 'panzoom'
    temp1 = np.zeros((512, 512), dtype=np.uint8); temp1[:, ::5] = 255
    speckle_image1 = scene.visuals.Image(temp1, parent=speckle_view1.scene, cmap='gray', vmin=0, vmax=255)
    speckle_view1.camera.set_range(margin=0); speckle_view1.camera.aspect = 300/300

    sat_label1 = scene.Label("Sat.: 0.00%", color='white', font_size=12); sat_label1.height_max = 60
    speckle_grid.add_widget(sat_label1, row=0, col=1)

    hist_view1 = speckle_grid.add_view(row=1, col=1)
    hist_view1.camera = 'panzoom'; hist_view1.camera.set_range(x=(0, 255), y=(0, 1.05))
    bins_edges = np.arange(257, dtype=np.float32); bins_centers = (bins_edges[:-1] + bins_edges[1:]) / 2.0
    hist_y1 = np.zeros(256, dtype=np.float32)
    hist_line1 = scene.visuals.Line(np.column_stack([bins_centers, hist_y1]), parent=hist_view1.scene, width=1, color='white')

    speckle_view2 = speckle_grid.add_view(row=1, col=2)
    speckle_view2.camera = 'panzoom'
    temp2 = np.zeros((512, 512), dtype=np.uint8); temp2[::5, :] = 255
    speckle_image2 = scene.visuals.Image(temp2, parent=speckle_view2.scene, cmap='gray', vmin=0, vmax=255)
    speckle_view2.camera.set_range(margin=0); speckle_view2.camera.aspect = 300/300

    sat_label2 = scene.Label("Sat.: 0.00%", color='white', font_size=12); sat_label2.height_max = 60
    speckle_grid.add_widget(sat_label2, row=0, col=3)

    hist_view2 = speckle_grid.add_view(row=1, col=3)
    hist_view2.camera = 'panzoom'; hist_view2.camera.set_range(x=(0, 255), y=(0, 1.05))
    hist_y2 = np.zeros(256, dtype=np.float32)
    hist_line2 = scene.visuals.Line(np.column_stack([bins_centers, hist_y2]), parent=hist_view2.scene, width=1, color='white')

    # Correlation
    corr_canvas = scene.SceneCanvas(keys='interactive', show=True, title='Correlation monitor', bgcolor='black', size=(600, 400))
    corr_grid = corr_canvas.central_widget.add_grid(margin=0)
    corr_grid.spacing = 2

    title_corr = scene.Label("Single (RED): 0.0000 / Double (BLUE): 0.0000", color='white', font_size=12)
    title_corr.height_max = 80
    corr_grid.add_widget(title_corr, row=0, col=0, col_span=2)

    yaxis = scene.AxisWidget(orientation='left', axis_label='Correlation', axis_font_size=8, axis_label_margin=60, tick_label_margin=15)
    yaxis.width_max = 100
    corr_grid.add_widget(yaxis, row=1, col=0)

    xaxis = scene.AxisWidget(orientation='bottom', axis_label='Time (min)', axis_font_size=8, axis_label_margin=40, tick_label_margin=20)

    xaxis.height_max = 80
    corr_grid.add_widget(xaxis, row=2, col=1)

    right_padding = corr_grid.add_widget(row=1, col=2, row_span=1)
    right_padding.width_max = 30

    corr_view = corr_grid.add_view(row=1, col=1)

    N_corr = 10
    xdata_corr1 = deque([0.0]*N_corr, maxlen=N_corr)
    xdata_corr2 = deque([0.0]*N_corr, maxlen=N_corr)

    ydata_corr1 = deque([0.0]*N_corr, maxlen=N_corr)
    ydata_corr2 = deque([0.0]*N_corr, maxlen=N_corr)

    line_corr1 = scene.visuals.Line(
        np.column_stack([np.array(xdata_corr1, dtype=np.float32),
                        np.array(ydata_corr1, dtype=np.float32)]),
        color='red', parent=corr_view.scene, width=1, name='Single'
    )
    line_corr2 = scene.visuals.Line(
        np.column_stack([np.array(xdata_corr2, dtype=np.float32),
                        np.array(ydata_corr2, dtype=np.float32)]),
        color='blue', parent=corr_view.scene, width=1, name='Double'
    )
    
    corr_view.camera = 'panzoom'

    xaxis.link_view(corr_view)
    yaxis.link_view(corr_view)

    grid_lines = scene.visuals.GridLines(color=(0.3, 0.3, 0.3, 0.5))
    corr_view.add(grid_lines)

    corr_view.camera.set_range(x=(0, N_corr), y=(0, 1))

    # Reservoir state
    states_canvas = scene.SceneCanvas(keys='interactive', show=True, title='Reservoir states monitor', bgcolor='black', size=(600, 400))
    states_grid = states_canvas.central_widget.add_grid(margin=0)
    states_grid.spacing = 2

    title = scene.Label("Reservoir states", color='white', font_size=12)
    title.height_max = 80
    states_grid.add_widget(title, row=0, col=0, col_span=2)

    yaxis = scene.AxisWidget(orientation='left', axis_label='Neuron', axis_font_size=8,
                             axis_label_margin=60, tick_label_margin=15)
    yaxis.width_max = 100
    states_grid.add_widget(yaxis, row=1, col=0)

    xaxis = scene.AxisWidget(orientation='bottom', axis_label='Time (min)', axis_font_size=8,
                             axis_label_margin=40, tick_label_margin=20)
    xaxis.height_max = 80
    states_grid.add_widget(xaxis, row=2, col=1)

    right_padding = states_grid.add_widget(row=1, col=2, row_span=1)
    right_padding.width_max = 30

    states_view = states_grid.add_view(row=1, col=1)
    states_view.camera = 'panzoom'

    xaxis.link_view(states_view)
    yaxis.link_view(states_view)

    grid_lines = scene.visuals.GridLines(color=(0.3, 0.3, 0.3, 0.5))
    states_view.add(grid_lines)

    T_states = 200
    N_states = 5
    xdata_states = deque([0.0]*T_states, maxlen=T_states)
    ydata_states = [deque([0.0]*T_states, maxlen=T_states) for _ in range(N_states)]

    colors_states = ['red', 'green', 'blue', 'orange', 'cyan']
    lines_states = []
    for i in range(N_states):
        line_temp = scene.visuals.Line(
            np.column_stack([np.array(xdata_states, dtype=np.float32),
                             np.array(ydata_states[i], dtype=np.float32) + i]),
            color=colors_states[i % len(colors_states)],
            parent=states_view.scene,
            width=1,
        )
        lines_states.append(line_temp)

    states_view.camera.set_range(x=(0, T_states), y=(-1, N_states+1))

    # Move windows
    mask_canvas.native.move(200, 50)
    speckle_canvas.native.move(200, 500)
    corr_canvas.native.move(1050, 50)
    states_canvas.native.move(1050, 500) 

    def update(_ev):
        # Drain queues quickly (avoid backlog)
        while not mask_queue.empty():
            mask = mask_queue.get_nowait()
            if mask is not None:
                mask_img.set_data(mask)
        mask_canvas.update()

        while not speckle_queue.empty():
            speckle = speckle_queue.get_nowait()
            if speckle is not None:
                speckle_image1.set_data(speckle)
                counts, _ = np.histogram(speckle, bins=256, range=(0, 255))
                h = counts.astype(np.float32); 
                if h.max() > 0: h /= h.max()
                hist_line1.set_data(np.column_stack([bins_centers, h]))
                hist_view1.camera.set_range(x=(0, 255), y=(0, 1.05))
                sat_pct = (np.count_nonzero(speckle >= 255) / speckle.size) * 100.0
                sat_label1.text = f"Sat.: {sat_pct:.2f}%"

        while not speckle2_queue.empty():
            speckle2 = speckle2_queue.get_nowait()
            if speckle2 is not None:
                speckle_image2.set_data(speckle2)
                counts2, _ = np.histogram(speckle2, bins=256, range=(0, 255))
                h2 = counts2.astype(np.float32)
                if h2.max() > 0: h2 /= h2.max()
                hist_line2.set_data(np.column_stack([bins_centers, h2]))
                hist_view2.camera.set_range(x=(0, 255), y=(0, 1.05))
                sat_pct2 = (np.count_nonzero(speckle2 >= 255) / speckle2.size) * 100.0
                sat_label2.text = f"Sat.: {sat_pct2:.2f}%"

        speckle_canvas.update()

        while not corr_queue.empty():
            minutes1, val1, minutes2, val2 = corr_queue.get_nowait()
            xdata_corr1.append(float(minutes1))
            ydata_corr1.append(float(val1))
            xdata_corr2.append(float(minutes2))
            ydata_corr2.append(float(val2))
            title_corr.text = f"Single (RED): {val1:.4f} / Double (BLUE): {val2:.4f}"

        line_corr1.set_data(np.column_stack([
            np.array(xdata_corr1, dtype=np.float32),
            np.array(ydata_corr1, dtype=np.float32)
        ]))
        line_corr2.set_data(np.column_stack([
            np.array(xdata_corr2, dtype=np.float32),
            np.array(ydata_corr2, dtype=np.float32)
        ]))

        corr_view.camera.set_range(x=(min(np.hstack((xdata_corr1, xdata_corr2))), max(np.hstack((xdata_corr1, xdata_corr2)))), y=(0, 1))

        # N_states-series
        while not states_queue.empty():
            minutes, arr = states_queue.get_nowait()
            xdata_states.append(float(minutes))
            for i, val in enumerate(arr):
                ydata_states[i].append(float(val))

        for i, l in enumerate(lines_states):
            l.set_data(np.column_stack([
                np.array(xdata_states, dtype=np.float32),
                np.array(ydata_states[i], dtype=np.float32) + i
            ]))

        states_view.camera.set_range(x=(min(xdata_states), max(xdata_states)), y=(-1, len(ydata_states)+1))

    timer = app.Timer(interval='auto', connect=update, start=True)
    app.run()
    

if __name__ == "__main__":

    try:
        # Important on Windows: keep GUI in main; workers in Processes -> I'm ignoring this advice :)
        mask_q = Queue()
        corr_q = Queue()
        spkl_q = Queue()
        spkl2_q = Queue()
        stts_q = Queue()

        p = Process(target=monitor, args=(mask_q, corr_q, spkl_q, spkl2_q, stts_q), daemon=True)
        p.start()

        # Dummy data
        for i in range(100):
            mask_q.put(np.random.uniform(0, 255, (1080, 1920)).astype(np.uint8))
            spkl_q.put(np.clip(np.random.normal(loc=130, scale=50, size=(300, 300)).astype(np.uint8)), 0, 255)
            spkl2_q.put(np.clip(np.random.normal(loc=130, scale=50, size=(300, 300)).astype(np.uint8)), 0, 255)
            corr_q.put((i, 0.75, i, 0.25))
            stts_q.put((i, np.random.random(5)))
            time.sleep(0.1)

        p.terminate()
        p.join(timeout=0.1)

    except KeyboardInterrupt:
        p.terminate()
        p.join(timeout=0.1)