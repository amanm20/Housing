# California Housing Price Prediction: An R Analysis Project

### Table of Content

- [Project Overview](#project-overview)
- [Data Source](#data-source)
- [Libraries Used](#libraries-used)
- [Exploratory Data Analysis](#exploratory-data-analysis-eda)
- [Model Evaluation and Selection](#model-evaluation-and-selection)
- [Results](#results)
  
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

We evaluate 3 different regression models to find the best one based on predictive performance.
- A full additive regression model including all predictors.
- A reduced model selected through forward selection, optimizing for predictors that contribute
most significantly to the outcome. 
- A LASSO regression model to enforce regularization, automatically selecting important features and shrinking coefficients of less important ones.

We begin by splitting the data into a training and test set (70-30 split). Then we build each of the 3 regression models and compute the RMSE on the test set based on out-of-sample predictions.

(I have only attached code snippets. To see detailed coding please check out the pdf document)
##### 1. Full additive MLR:
```R
housing_full_add <- lm(median_house_value ~ .,
data = training_housing)

housing_test_pred_full_add <- predict(housing_full_add,
newdata = testing_housing[, -5])

head(housing_test_pred_full_add)
```
<img width="895" alt="Screenshot 2024-03-21 at 6 35 24 AM" src="https://github.com/amanm20/Housing/assets/79020512/9abd0511-1474-4071-9078-2f330fbd9ed5">


##### 2. Forward selection:
```R
housing_forward_sel <- regsubsets(
x = median_house_value ~ .,
nvmax = 9,
data = training_housing,
method = "forward")

housing_fwd_summary_df <- tibble(
n_input_variables = 1:9,
RSS = housing_fwd_summary$rss,
BIC = housing_fwd_summary$bic,
Cp = housing_fwd_summary$cp)
```
<img width="437" alt="Screenshot 2024-03-21 at 6 39 05 AM" src="https://github.com/amanm20/Housing/assets/79020512/a0aca2ba-856b-4434-8190-8c4015613c5f">


##### 3. LASSO:
```R
Housing_cv_lambda_LASSO <- cv.glmnet(
x = model_matrix_X_train,
y = matrix_Y_train,
alpha = 1,
lambda = exp(seq(5, 12, 0.1)))

plot(Housing_cv_lambda_LASSO, main = "Lambda selection by CV with LASSO\n\n")

Housing_lambda_min_MSE_LASSO <- round(Housing_cv_lambda_LASSO$lambda.min, 4)

Housing_LASSO_min <- glmnet(
x = model_matrix_X_train,
y = matrix_Y_train,
alpha = 1,
lambda = Housing_lambda_min_MSE_LASSO)
```
<img width="903" alt="Screenshot 2024-03-21 at 6 36 09 AM" src="https://github.com/amanm20/Housing/assets/79020512/a333370d-6b4d-4c16-af87-ccd15ce129dd">

##### Comparing RMSE values to evaluate best model
Finally, we compare the RMSE of the 3 models to determine the best model based on prediction performance.

<img width="345" alt="Screenshot 2024-03-21 at 6 36 45 AM" src="https://github.com/amanm20/Housing/assets/79020512/545a560e-dcb8-4334-bdaf-ba7b5faf61cb">

The RMSE of the LASSO Regression model is 69983.56, which is the least among the 3 models, meaning the LASSO model has the best prediction performance!

### Results

We saw that median house price of a block increases with increase in median income as the two are strong positively correlated. The median house price also depends on the geographical location and proximity to ocean. It is higher in regions within close proximity to ocean/bay(s) that are on the coast.

We found that the RMSE of LASSO Regression (RMSE = 69983.56) is smaller than the RMSE of the other two models(Full Additive Regression Model RMSE: 70017.91 and Reduced Regression Model RMSE: 70017.34). This highlights the predictive accuracy of the LASSO Regression model in forecasting median house prices within our dataset. By shrinking the coefficients of less influential predictors towards zero, LASSO aids in model simplification and robustness against overfitting, resulting in a more generalizable model with enhanced predictive performance.
