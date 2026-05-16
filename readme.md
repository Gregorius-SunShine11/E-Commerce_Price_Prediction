# E-Commerce Price Prediction

This project aims to build a regression-based machine learning model for estimating item prices in e-commerce sites with limited features to simulate scenario when automated data scraping is unavailable. 

## Dataset

The train dataset contains 306,226 rows of entries, each having 26 features including the target feature (price). The test dataset contains 25,900 rows of entries with the same number of features, however only 300 entries spread across 3 days have most of the data filled, simulating scenario where the data is scraped manually, whereas the remaining entries only have data from the fully available features such as timestamp of scraping and shop/item IDs.


## Project Approach

This project follows a standard machine learning pipeline:

### 1. Data Preprocessing + Exploratory Data Analysis (EDA)

The preprocessing steps include:
- Handling missing values
- Encoding categorical variables
- Missing-features data augmentation

The exploratory data analysis performed consists of:
- Basic dataset statistics (count, mean, median, etc.)
- Feature Correlation Analysis
- Histogram Plot

### 2. Feature Engineering

Some of the features are dropped based on insights gathered from the EDA, such as:
- stock & normal_stock (only has zero or missing values)
- show_discount (high correlation with other independent features)

New features are also engineered, such as:
- Missing values flag (useful for tree-based models)
- Temporal data (day, hour, day of week, holiday flag)
- Item/shop-specific data (average price by item/shop ID, average discount by shop, etc.)

### 3. Model Selection

The following models are selected to be trained and compared:

| Model         | Parameters                                  |
|:--------------|:--------------------------------------------|
| Decision Tree | criterion='squared_error'                   |
| Random Forest | n_estimators=101, criterion='squared_error' |
| XGBoost       | eta=0.01, n_estimators=101, reg_alpha=1     |
| CatBoost      | learning_rate=0.01, n_estimators=101        |

These models are capable of dealing with non-linear relationships (shown in histogram plots) and handling missing values natively.

### 4. Model Training & Validation

The aforementioned models are trained with train dataset using 75%-25% train-validation split. The metrics used to validate the models are the following:
- Mean Absolute Percentile Error (MAPE, main metric)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared (R^2) score

## Results

The model with the best performance during validation is the Random Forest model with 5.18% MAPE. During the testing phase, the model is recalibrated using the anchor set, and after another validation using the anchor set, it had slightly higher MAPE of 7.02%.

## Code Instructions

The train dataset is available at the csv file 'ecommerce_price_prediction-train.csv', whereas the test dataset is available at the csv file 'ecommerce_price_prediction-test-3-days.csv'. You can follow this step-by-step instruction to run the project:

1. Run the notebook 'Model_Training_and_Validation.ipynb' for the model training phase. After running the notebook, it should result in three new files:
- best_model.pkl (the artifact of the trained model)
- brand_encoder.pkl (the artifact of the encoder used for 'brand' feature)
- ecommerce_price_prediction-train-preprocessed.csv (the preprocessed version of the train dataset)

2. Run the notebook 'Model_Testing_Live_Prediction.ipynb' to get the model prediction for test dataset, which is saved in the new csv file 'ecommerce_price_prediction-results.csv'.
