# Forecasting and evaluation methodology

## A forecast has an origin and a target

The **origin** is the last month whose actual quantity is available. The **target** is the month being forecast. A one-month horizon predicts the next month; a three-month horizon predicts three months ahead. The code fits a new model for each rolling origin and horizon using targets observed **no later than the origin**.

Random Forest uses 24 months of preceding demand to form lag and mean features. It also receives the item identifier, target calendar month, and horizon. Its target is a later observed quantity. The optional LSTM receives a 24-month sequence divided by that window's mean and learns a direct target-to-mean ratio for the selected horizon. Its prediction is multiplied by the item's latest window mean. This normalization lets one small example model learn across items with different scales. Neither model receives future actuals as features.

The **naive** baseline repeats the last observed month. The **seasonal naive** baseline uses the target month from the preceding year. These simple baselines matter because a complex model should earn its place through a fair comparison.

## Backtesting procedure

For each selected origin:

1. Limit all training examples and features to history available at that origin.
2. Fit the requested model separately for that origin and horizon.
3. Save its forecast before reading the target actual for scoring.
4. Match forecast and actual on item and target month.
5. If a benchmark CSV is supplied, restrict every model to the same item, origin, and target rows.

This is a retrospective simulation. A prospective production test, with predictions archived before outcomes occur, provides stronger evidence of real-world performance. Keep the two types of evidence distinct.

## Interpreting ±25% hit rate

The hit rule is `abs(forecast - actual) <= 0.25 * abs(actual)`. Report the numerator, denominator, aggregation level, horizon, item mix, months, and zero-actual rule along with any percentage. A claim such as “80% versus 65%” is hard to evaluate without those details. In particular, do not compare one method on all products with another on a selected subset.

WAPE instead divides **total absolute error** by **total absolute actual**. It is sensitive to large-volume items, while an unweighted hit rate gives every matched item-month row equal weight. Neither metric captures all operational costs. Overstock and shortage costs may be asymmetric.

## Limits of this example

- Its synthetic sample is designed to teach the workflow, not to validate performance on a real factory.
- The code requires complete monthly histories and does not handle product launches, discontinuations, stockouts, promotions, price changes, calendar effects beyond month, hierarchy reconciliation, or causal drivers automatically.
- The optional LSTM is a comparison example, not a recommended default. Small datasets often favor simpler models.
- Random Forest point forecasts do not provide calibrated uncertainty intervals here.
- A production deployment needs source-data governance, monitoring, retraining rules, access control, and human review for consequential decisions.
- Forecast quality can drift as demand changes. Re-evaluate against newly realized outcomes.

To extend the framework, first record a baseline and prospective predictions, then add a clearly defined feature with an availability date. Test that the feature would truly have been known at each origin. Only keep additional complexity if it improves the relevant out-of-sample measure on comparable rows.
