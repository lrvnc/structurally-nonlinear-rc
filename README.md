# Code: Optical Reservoir Computing with Structural Nonlinearity for Forecasting Chaotic Time Series

---

[![REPO DOI](assets/repo-doi-badge.svg)](https://doi.org/10.5281/zenodo.22229521)

This repository contains the base code that supports the work "Optical Reservoir Computing with Structural Nonlinearity for Forecasting Chaotic Time Series".

## Files

- `utils.py`: contains auxiliary functions, including the implementation of the Diebold-Mariano test.
- `encoding.py`: contains the code for the basket encoding binarization strategy.
- `visuals.py`: contains the code to monitor the masks sent to the DMD (misc).
- `optical_setup.py`: contains the code necessary to control the optical setup.
- `optical_setup.ipynb`: notebook containing tests executed during experiments (misc).
- `reservoir.py`: contains the reservoir computing framework code.
- `reservoir.ipynb`: notebook with an example of how to run the reservoir code.
- `mackey_glass.py`: contains the code to generate the train and test splits.
- `compile_raw_data.ipynb`: contains the code used to compile the predictions for further analysis.
- `figures.ipynb`: notebook containing the code used to generate the figures in the paper.

## Data format

The data needed to reproduce the analysis are stored in `results.joblib` (you can find it in Zenodo). Once loaded, it can be accessed as:

```
results = joblib.load(r"data\results.joblib")
results[key1][key2]
```

`key1` is the amount of training data used to produce the predictions. `key2` selects one of:

- `y_true`: the ground truth
- `y_lin`: predictions from the conventional approach
- `y_nonlin`: predictions from the structurally nonlinear approach
- `X_train`: the training data

For example, to retrieve all predictions from the structurally nonlinear approach obtained with 10k training points:

```
predictions = results[10000]['y_nonlin']
```

`predictions` has shape `(T, S, H)`, where `T` is the number of trials (14), `S` is the number of samples in the evaluation split (5000), and `H` is the prediction horizon (500).

## Citation

If you find this code useful, consider citing us:

```bibtex
@article{structurallyNonlinearRC,
  title   = {Optical Reservoir Computing with Structural Nonlinearity for Forecasting Chaotic Time Series (code)},
  author = {Venâncio, Leandro R and Dong, Jonathan and Bertolotti, Jacopo and Mounaix, Mickael},
  year    = {2026},
  doi     = {10.5281/zenodo.22229521}
}
```

## Contact

For questions, please contact any of the authors.
