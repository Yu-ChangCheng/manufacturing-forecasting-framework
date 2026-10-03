# A plain-language guide to the forecasting app

You do not need to write code to use the app. You need a CSV or Excel `.xlsx` file with dates and actual quantities. The app helps you put those columns into a monthly table, compare methods on past months, and download a future estimate. It reads the first sheet of an Excel workbook.

![Five steps from CSV to forecast](workflow.svg)

## 1. Open the app

Install Python 3.11 or newer. On Windows, double-click [`START_APP.bat`](../START_APP.bat). It creates a local environment, installs the app packages, and opens the browser. The first run needs internet access for the package installation. You can inspect the short batch file before running it.

If you prefer PowerShell, open it in the repository folder and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[app]"
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

The browser opens a page served on your computer. Keep PowerShell open while using it. To stop, press `Ctrl+C`. If your computer has more than one Python version, replace `python` in the first line with `py -3.13` or another installed version. The command-line workflow remains available in the [README](../README.md).

## 2. Learn with the fictional Random Forest example

Keep the default **See the Random Forest teaching example** option. It has six fictional items with a repeating yearly pattern and random noise. Click **Run the backtest and make a future forecast**. The default tests six past forecast dates; move the slider to **12** to test every month of 2025 and reproduce the [illustrated result](random-forest-demo.md). Random Forest performs well on this designed example because its inputs include the calendar month and recent demand. Its score does not predict how well it will perform on your data.

Then choose **Explore public car-parts example** to see a less regular public dataset. Keep the default items, one-month horizon, and six past forecast dates. Click the same run button. This small Monash car-parts sample from Hugging Face contains intermittent demand and some zeros that the source originally inserted for missing observations. Random Forest does not lead on the ±25% hit rate here; inspect both the metric table and chart. Read the [source and attribution note](public-data.md) before quoting any result.

![A backtest hides later actuals until scoring](backtest-explained.svg)

## 3. Prepare your own CSV

Choose **Use my own file**. A simple CSV can look like this:

```csv
month,item_id,quantity
2022-01-01,PART-A,105
2022-02-01,PART-A,98
2022-03-01,PART-A,113
```

The app also accepts a file with separate `year` and `month` columns. If you have individual daily sales or transaction rows, choose **A transaction or daily record** so the app sums rows for each item and month. If your file already has one item-month row, choose **One item in one month**; duplicates will be flagged instead of silently summed. If you do not distinguish items, choose **All rows are one total**.

Use actual sold or demanded **quantities**, not dollar revenue or existing forecasts, in the quantity column. Resolve returns, canceled orders, nonnumeric cells, and negative values before uploading. A stockout can hide true demand, so sold quantity may be lower than demand.

**Missing months require your decision.** The app stops if an item skips a month. Check the source. Only check the zero-fill box when that item was active and its true demand was zero. Do not use zero to represent an unknown observation, a product not yet launched, or an unavailable record. The downloaded prepared CSV shows exactly which monthly history the model used.

Each selected item needs at least 24 months of earlier history plus later months to test. A one-month-ahead backtest at six forecast dates needs at least 31 consecutive months. More history and more varied time periods make the comparison more useful. Selected items must end in the same month in this beginner interface.

## 4. Read the backtest

The app compares:

| Method | Plain meaning |
| --- | --- |
| Last month | Predict the same quantity as the latest known month. |
| Same month last year | Predict the quantity from the corresponding month a year earlier. |
| Random Forest | Learn patterns from preceding months and item IDs. It still uses only past information. |
| Your saved forecast (optional) | A forecast recorded before the actual quantity was known. |

The **±25% hit rate** counts item-month forecasts whose error is no more than one-quarter of the actual quantity. If actual demand is zero, only a zero forecast is a hit. **WAPE** adds all absolute errors and divides by all actual quantities; lower is better. It gives high-volume observations more influence than the hit rate. **Bias** shows the overall direction of error: positive means overforecasting, negative means underforecasting.

The methods can rank differently under these measures. Check the count of item-month forecasts, the item selection, the months tested, and the chart. A retrospective backtest is a simulation of past decisions. Keep actual future forecasts saved before outcomes happen if you want stronger evidence of real-world performance.

For a fair comparison with a sales team, expand **Optional: compare a forecast saved by your team**. Download its template. Each row needs the last month known when the forecast was made (`origin_month`), the month predicted (`target_month`), `item_id`, and `forecast_qty`. Both date columns use the first of the month, such as `2025-01-01`. The app scores all methods only on item, origin, and target combinations present in that saved file. A forecast reconstructed after actual sales became known is not a valid prospective comparison.

## 5. Download and review

Download the past predictions, future forecast, and plain-language report. The future forecast should be reviewed alongside known promotions, supply interruptions, launches, unusual orders, and business constraints. This prototype gives point estimates and does not calculate a reliable uncertainty interval or make inventory decisions for you.

For private company data, run the app on a trusted local computer with approval to use the data. Keep the original file and downloaded results out of the public Git repository. Hosting the app on a shared server changes who may have access to uploaded information.

## Common problems

| Message | What to do |
| --- | --- |
| Duplicate item and month | Confirm whether rows are transactions that should be summed or a duplicate data error. |
| Missing month | Check the source; explicitly fill zero only when zero demand is true. |
| Not enough months | Add earlier history, reduce the test period, or choose a shorter forecast horizon. |
| Different last months | Select items whose records end in the same month. |
| Invalid quantity | Remove text, blanks, negative returns, and infinity values from the quantity column. |
