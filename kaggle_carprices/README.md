# Kaggle car price competition
 Yandex practicum workshop-1 
 
## Task
Predict car selling price on standard Kaggle dataset.
The dataset consists of training and test files
The test file is unlabelled and prediction on it must be loaded on Kaggle for hidden estimation that assigns score

## Tools
- python
- Jupyter Notebook
- sklearn
- matplotlib

## Files
### Python modules
All notebooks based on common library of three modules:
- utils.py - basic operations with dataframes
- dataset.py - operations on "train-test" pair of dataframes (class DataSet)
- task.py - base class for estimating/submitting process  

### Notebooks
- validation.ipynb - for primary evaluation 
- cross_validation.ipynb - for cross validation
- submit.ipynb - for providing finally results
- data_analysis.ipynb - for features importance and other investigations on the model

## Approach
In the research, next approaches were tried:
- tuning hyperparameters of models
- reduction of cardinality both in categorical and quntitative features 
- upscale records of cheap sales  (some effect with OrdinalEncoder  but useless with TargetEncoder) 

### Process
Every testrun writes result to separate logfile with timestamp in the name,
 these logs stored in "log" (git-ignored ) catalog,
 best results may be transferred to "selected" (git-supported ) catalog

## Result
Best Kaggle score achieved on submission is 15.75,
 best score confirmed by crosvalidation is 17.1

## Todo
- log parser notebook  to get easily estimations from logs
- separate notebook for feature analysis 
- more algorithms of  data preparation
- CatBoost model research ( probed ,but results were disappointing )