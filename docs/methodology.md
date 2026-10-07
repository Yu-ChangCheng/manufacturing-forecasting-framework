# Forecasting and evaluation methodology

This document defines the forecasts, time boundaries, and metrics used in this repository. The equations describe this implementation; the [research references](references.md) give the background for Random Forests, rolling-origin evaluation, forecast metrics, and the public dataset. The ±25% threshold is a project choice, not a claim made by those papers.

## Forecast origin, target, and features

Let $y_{i,t}$ be the observed nonnegative quantity for item $i$ in month $t$. An **origin** $o$ is the last month with known actuals. A **horizon** $h$ is 1–12 months, giving target month $o+h$. Only data dated at or before $o$ may enter a forecast for that target.

The Random Forest feature vector $x_{i,o,h}$ contains an item identifier, target calendar month, horizon, selected lags, and rolling means. With `lag_1` defined as the most recent known quantity:

$$
\operatorname{lag}_{\ell}(i,o)=y_{i,o-\ell+1}, \qquad
\operatorname{mean}_{w}(i,o)=\frac{1}{w}\sum_{j=0}^{w-1}y_{i,o-j}.
$$

The command-line tool uses lags 1, 2, 3, 6, 12, and 24 and rolling windows 3, 6, and 12. The [workbench notebook](technical-workbench.md) lets you change them. The item identifier is one-hot encoded. The fitted forest of $B$ regression trees produces a nonnegative point forecast:

$$
\hat y_{i,o+h}=\max\!\left(0,\frac{1}{B}\sum_{b=1}^{B} T_b(x_{i,o,h})\right).
$$

The clip reflects a quantity that cannot be negative. The model receives no future actual quantity as a feature. The optional LSTM takes a 24-month sequence scaled by that window's mean and predicts a target-to-mean ratio for the chosen horizon; it is included as an experiment, not as the default recommendation.

## Baselines

The **last-month** baseline repeats $y_{i,o}$. For horizons up to 12 months, the **same-month-last-year** baseline uses $y_{i,o+h-12}$. These are forecasts available at origin $o$. A complex model is useful only if it improves a relevant out-of-sample comparison with such baselines.

## Two evaluation paths

**Guided app and command-line backtest.** For each selected origin and horizon, the tool forms training rows whose target labels were observed by that origin, fits a new model, makes forecasts, then scores them against later actuals. This is a rolling-origin retrospective test. When an external forecast CSV is supplied, every method is compared on the same item, origin, and target rows. Historical external forecasts should have been saved before the corresponding actuals were known.

**Technical workbench notebook.** The notebook reserves the last `TEST_ORIGINS` as a final holdout. Training labels must have target dates on or before the **first** test origin. Optional hyperparameter search uses earlier rolling validation folds only. It then fits **one model** on the pre-holdout training data and scores that fixed model across the holdout origins. Later holdout origins can use actuals that became known by those origins as lag inputs, but model parameters stay fixed during the holdout. After evaluation, a separate final model may be trained on all observed history for future use. Changing settings after looking at the holdout score uses up that holdout as an unbiased test; evaluate on new later data before making a strong performance claim.

For either path, publish the item and month coverage, forecast horizon, number of origins, model settings, baseline definitions, and the date each input would have been available. Tashman's [rolling-origin review](references.md#rolling-origin-evaluation) motivates evaluating forecasts at multiple historical decision points. A prospective test with forecasts archived before outcomes occur is stronger evidence of operating performance than a retrospective simulation.

## Metrics on matched rows

For $N$ matched item-origin-target rows, let $y_j$ be the actual quantity and $\hat y_j$ the forecast. This project's **within ±25% hit rate** is:

$$
\operatorname{Hit}_{25}=\frac{1}{N}\sum_{j=1}^{N}
\mathbf{1}\!\left\{|\hat y_j-y_j|\leq 0.25|y_j|\right\}.
$$

Each row gets equal weight. If $y_j=0$, only an exact zero forecast counts as a hit (subject to a tiny numerical tolerance in the code). A percentage such as “80% versus 65%” needs its two hit counts, common denominator, same item-month rows, horizon, time period, and zero rule to be meaningful.

**Weighted absolute percentage error (WAPE)** and **signed bias ratio** are:

$$
\operatorname{WAPE}=\frac{\sum_{j=1}^{N}|\hat y_j-y_j|}{\sum_{j=1}^{N}|y_j|},
\qquad
\operatorname{Bias}=\frac{\sum_{j=1}^{N}(\hat y_j-y_j)}{\sum_{j=1}^{N}|y_j|}.
$$

WAPE gives high-volume rows more influence than the unweighted hit rate. Positive bias means aggregate overforecasting; negative bias means underforecasting. Both ratios are undefined when the total absolute actual quantity is zero, and the tool leaves them blank. No single metric describes every planning cost. Hyndman and Koehler's [forecast-accuracy review](references.md#forecast-accuracy-measures) is useful context for metric choice; it does not establish this repository's ±25% rule.

## Limits of this example

- The synthetic sample is designed to teach the workflow, not validate performance on a real factory. The public car-parts subset is also a teaching sample, not an independent adoption study.
- The code requires complete monthly histories. It does not automatically handle product launches, discontinuations, stockouts, promotions, price changes, hierarchy reconciliation, or causal drivers beyond the target month.
- Intermittent demand and zero months deserve separate analysis. A good WAPE can coexist with a weaker hit rate, and vice versa.
- Random Forest point forecasts do not provide calibrated uncertainty intervals here. Overstock and shortage costs may be asymmetric.
- Production use needs source-data governance, monitoring, retraining rules, access control, and human review. Forecast quality can drift as demand changes.

To extend the framework, first record a baseline and prospective predictions. Add a feature only if its value would truly have been known at each origin. Keep added complexity when it improves the chosen out-of-sample measure on comparable rows.
