# NBA Player Longevity Prediction — Feature Engineering Pipeline

A rigorous feature engineering, target validation, and data transformation pipeline built using Python and Scikit-Learn. This framework processes raw NBA rookie box-scores into a standard-scaled, independent, and robust dataset optimized for machine learning algorithms.

## 🎯 Prediction Goal
The analytical objective of this project is to optimize predictive inputs determining whether an NBA rookie's professional career will survive or exceed five years (`target_5yrs`). 

---

## 🛠️ Feature Engineering Workflow

### 1. Data Cleansing & Data Leakage Prevention
* **Target Isolation**: `target_5yrs` is isolated as the formal binary dependent class label.
* **Target Class Balance Check**: An explicit evaluation step maps the target's frequency. The dataset contains a stable class distribution (~62.1% positive class, ~37.9% negative class), verified using visual bar plots exported during execution.
* **Identifier Filtering**: High-cardinality nominal parameters (`name`) and index references (`Unnamed: 0`) are removed to prevent data leakage and overfitting on row order.
* **Outlier-Robust Imputation**: Structural nulls are safely imputed using feature-level **median** values, safeguarding structural integrity against skewing from extreme rookie outliers.

### 2. Composite Metric Generation
To separate raw runtime metrics from technical output density, two major metric transformations are executed:
* **Points Per Minute ($PPM$)**: Formulated as:
  $$PPM = \frac{\text{pts}}{\text{min} + 1e-5}$$
  Grades scoring efficiency regardless of playing role (starters vs. bench players).
* **Player Efficiency Index ($PER_{\text{simple}}$)**: Evaluated as a linear utility combination tracking cumulative positive contributions against possession waste:
  $$\text{Efficiency} = (\text{pts} + \text{reb} + \text{ast} + \text{stl} + \text{blk}) - [(\text{fga} - \text{fgm}) + (\text{fta} - \text{ftm}) + \text{tov}]$$

### 3. Multicollinearity Assessment & Feature Drop Rationale
Basketball performance indicators carry systemic mathematical redundancy. Using an automated correlation threshold filter ($|r| > 0.85$), collinear parameters were dropped to maximize coefficient interpretability.
* **Dropped Features**: `fgm`, `fga`, `ftm`, `fta`, `oreb`, `dreb`, `pts`, `3pa`.
* **Retained Features**: `gp`, `min`, `fg`, `3p_made`, `3p`, `ft`, `reb`, `ast`, `stl`, `blk`, `tov`, along with the engineered `Points_Per_Minute` and `Player_Efficiency` parameters.

### 4. Feature Transformation & Scaling
* **Methodology**: **`StandardScaler`** (Z-score normalization via Scikit-Learn) is explicitly applied to the imputed, reduced feature array.
* **Rationale**: Normalization scales columns to $\mu=0, \sigma=1$. This step ensures uniform feature weightings, allowing distance-based and regularized linear classifiers (e.g., Logistic Regression, SVMs) to converge correctly without scaling biases.

---

## 📊 Resulting Artifacts
* **Processed Dataset**: `nba_players_feature_engineered.csv` (Standard-scaled predictors + unscaled target variable).
* **Diagnostic Plot Visuals**: 
  * `target_class_balance.png` (Class distribution bar plot).
  * `correlation_matrix_before.png` (Initial collinear landscape).
  * `correlation_matrix_after.png` (Final pruned independent feature matrix).

---

## 🚀 Execution Guide
1. Place the raw data file `nba-players.csv` into your working directory.
2. Open the newly cleaned script file or run your environment cells in sequence.
3. Select **Run All Cells** to instantly output the clean dataset matrix and complete diagnostic plots.
