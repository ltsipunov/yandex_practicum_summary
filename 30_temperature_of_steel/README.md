#  Steel Temperature prediction
Yandex Practicum final project

## Task
Process of  steel production  includes up to 15 heating cycles with different alloy additions.
- The model must predict temperature of steel for a given  sequence of operations.
- The training set contains about 3000 samples 
- Metric MAE <=8 required.

## Tools
- Jupyter Notebook
- numpy
- pandas
- sklearn
- matplotlib
- seaborn
- sqlalchemy
- lightgbm


## Files
temperature_of_steel.ipynb - main project file

## Results
### preanalysis.
- About 700 samples where classified as irrelevant and dropped.
- More than 100 features analysed for importance and correlation 
-- most of rare and  correlated features were dropped 
-- new artifical features generated to get new dataset good enaough to apply Linear Regression
### models
- several models with different algorithms(Regression,Gradient Boosting , MLP)  trained and compared.
- with SGDRegressor acieved the best result with MAE=5.8
- short analysis of features importance applied

## TODO
- check other feature generation , based on times BEFORE last heating 