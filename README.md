# Machine Learning Pipeline Project: 200+ Financial Indicators Stock Performance Prediction

This project implements an end-to-end Machine Learning pipeline for the predictive classification of US stock market performance based on corporate financial statements (SEC 10-K filings).

The project is developed for the **Machine Learning** course (SUPSI) in compliance with the guidelines specified in `Progetto.pdf`.

---

## 1. Dataset Description and Business Context

### Data Source
The dataset collects over 200 financial indicators for publicly traded companies on US stock markets (NYSE, NASDAQ, AMEX) over the five-year period **2014–2018**. The data was scraped via the [Financial Modeling Prep API](https://financialmodelingprep.com/developer/docs/) and originates directly from annual 10-K filings submitted to the SEC.

### Modeling Objective
The objective is to determine whether fundamental financial statement metrics from fiscal year $T$ carry sufficient predictive signal to forecast stock price performance in the following year $T+1$.

- **Machine Learning Problem**: Supervised binary classification.
- **Target Variable (`Class`)**:
  - `Class = 1`: The stock price records a positive return in the subsequent calendar year ($\text{Price Var} > 0\%$). Represents stocks to buy at the beginning of the year and sell at year-end.
  - `Class = 0`: The stock price records zero or negative return in the following year ($\text{Price Var} \le 0\%$).

### Data Dimensions
- **Volume**: ~3,800–4,300 companies per year, totaling over 20,000 observations across all 5 years (2014–2018).
- **Feature Space**: 225 original columns, consisting of:
  - Company Ticker (unique identifier).
  - Categorical feature `Sector` (e.g., *Technology, Healthcare, Financial Services, Industrials, Consumer Defensive*, etc.).
  - Quantitative accounting features (Balance Sheet, Income Statement, Cash Flow Statement, growth indicators, and valuation multiples such as P/E, P/B, EV/EBITDA, Debt/Equity, ROE, FCF).
  - Target leakage feature `[YEAR] PRICE VAR [%]` (strictly removed).
  - Binary target `Class`.

---

## 2. Temporal Validation Methodology ("Out-of-Time")

To ensure scientific rigor and prevent **temporal data leakage** (look-ahead bias) and cross-sectional contamination:
1. **Training & Validation Set (Years 2014–2017)**:
   - The first four historical years (~17,500 observations) are allocated for model training, preprocessing fitting, and cross-validation.
   - Hyperparameter tuning employs a temporal/rolling validation scheme (`TimeSeriesSplit` or annual split folds), respecting the forward arrow of time.
2. **Final Out-of-Sample Test Set (Year 2018)**:
   - The final available year (~4,300 observations) is strictly sequestered as a blind benchmark.
   - It is never exposed during preprocessing or hyperparameter tuning, closely replicating the real-world scenario of an investor making decisions at the start of 2018.

---

## 3. Feature Selection with LASSO ($L_1$ Regularization)

Given the high dimensionality (~220 features), high multicollinearity among accounting metrics, and statistical noise, feature selection is performed using **LASSO (Logistic Regression with $L_1$ penalty)**:

### Why LASSO?
- **Sparsity**: Unlike $L_2$ (Ridge) penalties, $L_1$ drives non-informative and redundant coefficients exactly to zero, compressing the space into a parsimonious set of key drivers.
- **Computational Efficiency**: Reduces dimensionality for distance-sensitive models (such as KNN) and accelerates grid search tuning.
- **Interpretability**: Enables inspection of specific accounting indicators that carry the strongest predictive weights.

### Operational Calibration and Selection Results
- **Initial Underfitting Challenge**: An overly aggressive initial $L_1$ penalty combined with an average-threshold filter collapsed the feature space to only 7 variables, risking severe accounting information loss across essential dimensions.
- **Adopted Solution**: Utilization of the stable `liblinear` coordinate descent solver, configuring `max_features=50` ranked by absolute coefficient magnitude.
- **Outcome**: Compression from **229 transformed features down to 50 selected features (78.2% dimensionality reduction)**, striking the optimal balance between noise suppression and comprehensive financial coverage.

### Economic and Macroeconomic Alignment of Selected Weights
Inspecting the estimated LASSO coefficients over the 2014–2017 training period reveals strong consistency with the macroeconomic reality of the era:
1. **Heavily penalized Energy sector (`Sector_Energy = -0.497`)**: Accurately mirrors the historic 2014–2016 crude oil crash, where WTI prices plunged from over \$100 to under \$30 per barrel, causing widespread distress and sell-offs across US energy firms.
2. **Premium on defensive and dividend-yielding sectors (`Sector_Utilities = +0.714`, `Sector_Financial Services = +0.380`, `Sector_Real Estate = +0.329`)**: Amid the Federal Reserve's Zero Interest Rate Policy (ZIRP), market participants actively engaged in "yield hunting," favoring stable cash flows and predictable distributions from utilities and REITs.
3. **Solid microeconomic fundamentals**: Predictors such as `3Y Dividend per Share Growth (+0.127)`, `5Y Dividend per Share Growth (+0.125)`, `10Y Net Income Growth (+0.082)`, and `Gross Margin (+0.030)` display stable positive coefficients. This confirms that companies demonstrating durable competitive advantages (*economic moats*) and consistent real cash return to shareholders have a significantly higher likelihood of stock price appreciation.

### Known Trade-offs and Limitations (Critical Discussion)
- **Collinear Features**: When pairs or clusters of features exhibit near-perfect correlation (e.g., `EBIT` vs `Operating Income`), LASSO tends to select one arbitrarily and set the others to zero.
- **Linearity Assumption**: Being a linear regularization method, LASSO may penalize non-linear relationships (e.g., inverted U-shaped relationships around optimal corporate leverage).
- **Scale Sensitivity**: Requires robust scaling (`StandardScaler` / `RobustScaler`) to prevent extreme financial statement outliers from distorting coefficient shrinkage.

---

## 4. Repository Structure

```text
Project/
├── data/
│   ├── 2014_Financial_Data.csv
│   ├── 2015_Financial_Data.csv
│   ├── 2016_Financial_Data.csv
│   ├── 2017_Financial_Data.csv
│   └── 2018_Financial_Data.csv
├── Financial_Stock_Classification_Pipeline.ipynb  # Main end-to-end notebook
└── README.md                                     # Methodological documentation
```
