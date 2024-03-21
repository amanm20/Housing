# California Housing Price Prediction: An R Analysis Project

### Table of Content

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Libraries Used](#libraries-used)
- [Exploratory Data Analysis](#exploratory-data-analysis-eda)
- [Model Evaluation and Selection](#model-evaluation-and-selection)
- [Results](results)
  
### Project Overview 
This data analysis project leverages R to analyze and predict median housing prices across California, utilizing a dataset comprising various predictors such as median income, ocean proximity, geographical coordinates, etc. Through comprehensive data analysis, including exploratory visualization, regression modeling, and feature selection, we discover the factors driving housing prices in California. Our goal is to provide a robust model for accurate price predictions, offering valuable insights for the real estate market.

### Data Source
The primary dataset used for this analysis can be found here [Housing dataset](https://www.kaggle.com/datasets/camnugent/californiahousing-prices). It contains information about 20640 blocks in California with their median house price, along with 10 descriptive variables

### Libraries Used
- tidyverse: Utilized for data manipulation and creating visualizations.
- glmnet: Used for building LASSO regression models to identify key predictors of housing prices with regularization.
- leaps: Used for performing model selection, helping to identify the most predictive subset of variables for median house value.
- corrplot: Applied to visualize correlations between predictors, aiding in the detection and handling of multicollinearity.
- fastDummies: Converts categorical variables into dummy variables, essential for including categorical predictors in regression models.
- car: Provides diagnostic tools for linear regression, checking for multicollinearity through variance inflation factors.
- ggplot2: Creates detailed and informative visualizations to explore the relationship between house prices and various predictors.

### Exploratory Data Analysis EDA
EDA involved exploring the median house price and seeing how it varies with different predictor variables.

##### 1. Median house price vs Median income of the block
![image](https://github.com/amanm20/Housing/assets/79020512/7330db70-d2f0-47ef-9caa-a773ea86720e)

The median house value of the block shows a stong positive correlation with the median income. We also notice that the median house value reaches a cap at about 500k USD. 

##### 2. Median house price vs Ocean proximity
![image](https://github.com/amanm20/Housing/assets/79020512/76edd696-a572-4874-b17d-e60c756cf72e)

The average median house value differs based on Ocean proximity. It is the highest in the Islands followed by regions with close proximity to water bodies and then the Inland.

##### 3. Median house price by geographical coordinates
![image](https://github.com/amanm20/Housing/assets/79020512/ed7afb96-bff5-477c-9d8b-54f7c6804b6b)

The median house value is highest near the coast and decreases as we move Inland. This aligns with our previous results based on ocean proximity.

### Model Evaluation and Selection

Evaluate 3 different types of regression models to find the best one based on predictive performance. begin by splitting data into training and test set (70-30). Then build each of the 3 regression models and compute the RMSE on the test set based on out-of-sample predictions. compare the 3 models based on RMSE to find which is the best based on predictive performance. 
```R

```

### Results
