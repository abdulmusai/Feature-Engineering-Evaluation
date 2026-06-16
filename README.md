# Feature-Engineering-Evaluation
In this project, you will analyze an NBA player performance dataset using Python and pandas to engineer features for a machine learning model.
NBA Player Longevity Prediction — Feature Engineering Pipeline
A rigorous feature engineering, target isolation, and multicollinearity reduction pipeline built using Python and pandas to transform raw NBA rookie performance box-scores into a clean, independent, and robust predictive dataset.
 Prediction Goal
The analytical objective of this project is to optimize predictive inputs determining whether an NBA rookie's professional career will survive or exceed five years (`target_5yrs`). Preparing sports performance indicators requires scaling and mapping combinations of cumulative and efficiency-driven statistics.
Feature Engineering Workflow
### 1. Data Cleansing & Data Leakage Prevention
* **Target Isolation**: `target_5yrs` is separated as the formal binary dependent class label.
* **Identifier Filtering**: High-cardinality nominal parameters (`name`) and arbitrary file index references (`Unnamed: 0`) are removed to ensure model decisions map purely to performance metrics rather than identifier strings.
* **Outlier-Robust Imputation**: Structural nulls are safely imputed using the median values of corresponding metrics, preserving demographic and distribution integrity without introducing downstream bias.
### 2. Composite Metric Generation
To separate raw runtime parameters from technical output density, two major metric transformations are executed:
* **Points Per Minute ($PPM$)**: Formulated as $\frac{\text{pts}}{\text{min}}$ to accurately grade scoring efficiency regardless of whether a player starts the game or comes off the bench.
* **Player Efficiency Index ($PER_{\text{simple}}$)**: Calculated as:
  $$\text{Efficiency} = (\text{pts} + \text{reb} + \text{ast} + \text{stl} + \text{blk}) - [(\text{fga} - \text{fgm}) + (\text{fta} - \text{ftm}) + \text{tov}]$$
  This wraps volume counts and turnover/shooting mistakes into a single predictive feature tracking individual utility.
### 3. Multicollinearity Assessment & Feature Drop Rationale
Basketball features carry systemic mathematical redundancy (e.g., points, field goals made, and field goals attempted are highly collinear). High multicollinearity inflates feature weight variance and blocks model interpretation.
Using an automated correlation matrix threshold filter ($|r| > 0.85$), the following strategic actions were completed:
* **Dropped Features**: `fgm`, `fga`, `ftm`, `fta`, `oreb`, `dreb`, `pts`, `3pa`.
* **Retained Features**: `gp`, `min`, `fg`, `3p_made`, `3p`, `ft`, `reb`, `ast`, `stl`, `blk`, `tov`, along with the newly engineered `Points_Per_Minute` and `Player_Efficiency` features.
## 📊 Resulting Dataset Structure
The engineered and cleared array shrinks column count down to independent, high-signal performance components.
* **File Produced**: `nba_players_feature_engineered.csv`
* **Artifacts Created**: `correlation_matrix_before.png`, `correlation_matrix_after.png`
## 🚀 Execution Guide
1. Put the raw file `nba-players.csv` in your working directory.
2. Launch the Jupyter Notebook: `nba_feature_engineering.ipynb`.
3. Select **Run All Cells** to produce the diagnostic plots and output dataset.	
