# House Price Regression & Heart Disease Classification

Supervised learning in scikit-learn on two public datasets:

- **Regression:** predict house sale prices from the [Ames Housing dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) (1,460 homes, 81 columns).
- **Classification:** predict heart disease from the [Cleveland Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) (297 patients, 13 features).

Each part goes through the same steps: explore the data, train a baseline model, try a second model, select features by correlation with the target, and compare train and test scores to check for overfitting.

**Tech:** Python · pandas · NumPy · scikit-learn · Matplotlib

Originally completed for CPSC 254 (Applied AI) at Cal State Fullerton.

## Results

| Task | Model | Test score |
|---|---|---|
| Ames Housing | Linear Regression, all numeric features | R² 0.818 |
| Ames Housing | Decision Tree, all numeric features | R² 0.702 (train R² 1.000: overfits) |
| Ames Housing | Decision Tree, 8 selected features | R² 0.820 |
| Heart Disease | Decision Tree, all features | 78.3% accuracy (train 100%: overfits) |
| Heart Disease | Decision Tree, 8 selected features, `max_depth=4` | **85.0% accuracy** |
| Heart Disease | Random Forest, 8 selected features | 83.3% accuracy |

Main takeaways:

- An unconstrained Decision Tree memorizes the training set on both datasets (perfect train score, much lower test score).
- Keeping only the features most correlated with the target, plus limiting tree depth, closed most of that gap. On the heart disease data, test accuracy went from 78.3% to 85.0%.
- The test sets are small (292 homes and 60 patients), so a single 80/20 split gives a rough estimate, not a precise one.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook house-price-heart-disease.ipynb
```

The datasets are not included. Download them from the links above and save them next to the notebook as `ames_housing.csv` (the Kaggle `train.csv`) and `heart_disease.csv` (the cleaned Cleveland data with a `condition` target column).
