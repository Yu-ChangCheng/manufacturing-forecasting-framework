# Research and data references

These sources explain the methods and provenance behind this educational implementation. They do **not** validate this repository's demonstration scores or imply that its authors studied, endorsed, or used this software. The software has its own [citation file](../CITATION.cff).

## Random Forests

Breiman, L. (2001). Random Forests. *Machine Learning, 45*, 5–32. [https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

This is the original Random Forest paper. The framework uses scikit-learn's `RandomForestRegressor` with lag, rolling-mean, item, and calendar features. The paper does not establish a particular accuracy level for this data or this implementation.

## Rolling-origin evaluation

Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An analysis and review. *International Journal of Forecasting, 16*(4), 437–450. [https://doi.org/10.1016/S0169-2070(00)00065-0](https://doi.org/10.1016/S0169-2070(00)00065-0).

The review discusses out-of-sample evaluation and rolling origins. Our app and CLI refit at each test origin; the notebook instead tunes on earlier folds and holds out later origins for one fixed model. [Methodology](methodology.md#two-evaluation-paths) specifies both protocols.

## Forecast accuracy measures

Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of forecast accuracy. *International Journal of Forecasting, 22*(4), 679–688. [https://doi.org/10.1016/j.ijforecast.2006.03.001](https://doi.org/10.1016/j.ijforecast.2006.03.001).

This review explains limitations of common error measures and proposes MASE. We cite it as background for reporting more than one score and checking zero-demand cases. **The ±25% hit rule, its threshold, and the specific WAPE and bias definitions in this project are our stated evaluation choices, not results attributed to this paper.**

## Public demonstration data

Godahewa, R., Bergmeir, C., Webb, G. I., Hyndman, R. J., & Montero-Manso, P. (2021). Monash Time Series Forecasting Archive. *NeurIPS Datasets and Benchmarks 2021*. [Paper](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/eddea82ad2755b24c4e168c5fc2ebd40-Paper-round2.pdf).

The archive includes the Car Parts Sales dataset. This repository contains only a selected five-series teaching subset prepared from its [Hugging Face copy and data card](public-data.md); see that page for transformation notes and license details. The paper documents the archive, not our demo's model performance.

## Implementation tutorial

scikit-learn developers. [Lagged features for time series forecasting](https://scikit-learn.org/stable/auto_examples/applications/plot_time_series_lagged_features.html). Accessed 2026-10-07.

This official example illustrates lag construction and the need for a time-aware evaluation. Its model and exact split differ from ours; our own code and [methodology](methodology.md) define this repository's behavior.
