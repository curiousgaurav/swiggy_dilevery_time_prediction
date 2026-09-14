# Swiggy Delivery Time Prediction - Model Accuracy Report

## Objective

This project predicts food delivery time in minutes using Swiggy delivery dataset features such as rider age, rating, traffic density, weather, delivery distance, festival, city, and order time.

## Dataset used

- Source: `data/raw/swiggy.csv`
- Target column: `Time_taken(min)`
- Data cleaning pipeline: `scripts/data_clean_utils.py`
- Train/test split: 80/20 with `random_state=42`

## Evaluated models

The following models were benchmarked on the same cleaned dataset:

- LinearRegression
- RandomForestRegressor
- LGBMRegressor
- StackingRegressor

## Evaluation metrics

Lower is better for MSE, RMSE, and MAE.
Higher is better for R².

| Model                 |     R² |     MSE |   RMSE |    MAE | Summary               |
| --------------------- | -----: | ------: | -----: | -----: | --------------------- |
| StackingRegressor     | 0.8419 | 13.5316 | 3.6785 | 2.9743 | Best overall accuracy |
| LGBMRegressor         | 0.8395 | 13.7318 | 3.7056 | 2.9921 | Very close to best    |
| RandomForestRegressor | 0.8321 | 14.3652 | 3.7901 | 3.0453 | Good and compact      |
| LinearRegression      | 0.6167 | 32.8037 | 5.7275 | 4.5752 | Weakest               |

## Best model

### Best accuracy model: StackingRegressor

Why it is best:

- Highest R² = 0.8419
- Lowest MSE = 13.5316
- Lowest RMSE = 3.6785
- Lowest MAE = 2.9743

This is the strongest model for predictive accuracy.

## Best free-tier deployment model

The original stacking model is too large for free-tier hosting on Render due to memory constraints. For deployment, a smaller and lighter model is needed.

### Recommended free-tier deployment model: LightGBMRegressor

Why:

- R² = 0.8395, very close to the best model
- MSE = 13.7318
- RMSE = 3.7056
- MAE = 2.9921
- Much smaller than the stacking model and suitable for free hosting

### Alternative free-tier option: RandomForestRegressor

Why:

- Still good performance: R² = 0.8321
- Smaller than stacking and still acceptable for deployment
- Simpler and easy to run in constrained memory environments

## Tradeoff summary

- Max accuracy: StackingRegressor
- Best practical free-hosting choice: LightGBMRegressor
- Simpler low-memory choice: RandomForestRegressor

## Final recommendation

For production and deployment on free hosting:

- Use the compact LightGBM model or compact RandomForest model.
- Keep the strong stacking model in the repository as the benchmark / best-performance model for comparison.

The project should treat these as two states:

1. High-accuracy research model: StackingRegressor
2. Free-hosting deployment model: LightGBMRegressor or RandomForestRegressor

## Verified local evidence

The latest local validation for the free-tier compact model produced:

- Model size: 17,164,643 bytes
- Sample prediction: 19.8149710922185

This confirms that the lightweight runtime model works and is compact enough for free-tier deployment.
