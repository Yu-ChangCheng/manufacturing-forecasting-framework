# Public car-parts example: source and limits

The file [`examples/hf_car_parts_sample.csv`](../examples/hf_car_parts_sample.csv) contains **five selected series, 51 months each, from January 1998 through March 2002**. It is a teaching subset of the **Car Parts Dataset (without Missing Values)**, hosted in the [AutonLab Time Series Pile on Hugging Face](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/blob/main/forecasting/monash/car_parts_dataset_without_missing_values.tsf). The underlying [Monash Time Series Forecasting Repository](https://huggingface.co/datasets/Monash-University/monash_tsf) lists a CC BY 4.0 license. The [original Zenodo record](https://zenodo.org/records/3994911) describes 2,674 monthly car-parts series and credits Godahewa, Bergmeir, and Webb. Please retain this attribution when reusing the subset.

The source `.tsf` file says **the original dataset contained missing values and those were replaced by zeros**. A zero in this example may therefore be an actual zero sale or an imputed missing observation. This limits conclusions about demand and especially the ±25% hit rate. The five series were chosen by the number of nonzero months, then total quantity, to make the charts easier to read. **No forecasting result was used to choose them.** The subset is not a random sample of all 2,674 series and must not be presented as a benchmark result for the full dataset.

The source file is pinned to Hugging Face revision `ea89753da2b451928436adb333c7a2e892461c7d`, SHA-256 `4308d4c75319f14ea3cfd853f5e7805d2f0c2eeb9a9a77e3ff517a8979c0a4f7`. To reproduce the CSV, download [the pinned `.tsf` source](https://huggingface.co/datasets/AutonLab/Timeseries-PILE/resolve/ea89753da2b451928436adb333c7a2e892461c7d/forecasting/monash/car_parts_dataset_without_missing_values.tsf) to `outputs/hf_car_parts_source.tsf`, then run:

```powershell
python scripts/prepare_hf_example.py --source outputs/hf_car_parts_source.tsf
```

The converter checks the source hash. The source download is in `outputs/`, which Git ignores. The derived 255-row CSV is committed so a new user can run the app without a network connection.

For an introductory one-month-ahead, six-origin backtest on this selection, the last-month baseline had a 33.3% ±25% hit rate and 96.8% WAPE; Random Forest had a 23.3% hit rate and 86.3% WAPE. The two measures rank them differently. This is one short retrospective illustration, **not** an estimate of future performance, a result from an employer, or support for any claimed company comparison.
