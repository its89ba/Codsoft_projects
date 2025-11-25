# Codsoft_Projects
Contains 3 project:
- Iris flower Classification
- Sales prediction using Advertising Data
- Movie rating Prediction
  
##  🗂️Iris Flower Classification
A comprehensive machine learning project for classifying iris flowers into three species based on their morphological features. This project demonstrates the complete workflow from data loading and exploration to model training and evaluation using multiple classification algorithms.

### Project Overview
This project uses the famous Iris dataset to build and compare multiple machine learning models for flower classification. The implementation includes data visualization, exploratory data analysis, and performance evaluation of various classifiers. The project showcases the entire machine learning pipeline from data preprocessing to model deployment.

###  Dataset
#### Source
**Iris.csv** from [Kaggle](https://www.kaggle.com/datasets/uciml/iris)

The project uses the classic Iris dataset from scikit-learn, containing:
- 150 samples of iris flowers
- 4 features for each sample:
  Sepal length (cm)
  Sepal width (cm)
  Petal length (cm)
  Petal width (cm)
- 3 target classes:
  Setosa (0)
  Versicolor (1)
  Virginica (2)

### Technologies Used
- Python 3
- NumPy - Numerical computations
- Pandas - Data manipulation and analysis
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- Scikit-learn - Machine learning algorithms and utilities
- Jupyter Notebook - Interactive development environment

### Machine Learning Models Implemented
The project implements and compares the following classification algorithms:
- Logistic Regression
- Support Vector Machine (SVC)
- Decision Tree Classifier
- Random Forest Classifier
- K-Nearest Neighbors (KNN)

###  Project Structure
#### 1. Data Loading and Preparation
- Load the Iris dataset from scikit-learn
-  Separate features and target variables
-  Create DataFrame for better data handling
-  Dataset overview and basic statistics

#### 2. Exploratory Data Analysis (EDA)
-  Comprehensive dataset analysis
-  Statistical summary of features
-  Feature correlation studies

#### 3. Data Visualization
- Scatter Plots:
  Sepal length vs Sepal width
  Petal length vs Petal width
- Box Plots: Feature distribution by species
- Correlation Heatmap: Feature relationships
- Multi-panel visualization for comprehensive insights

#### 4. Model Training Pipeline
- Train-test split (80-20 ratio)
- Feature scaling using StandardScaler
- Multiple classifier implementation
- Hyperparameter tuning and optimization

#### 5. Model Evaluation
- Accuracy scores comparison
- Detailed classification reports
- Confusion matrix analysis
- Performance metrics for each algorithm

###  Key Insights
Clear Species Separation: The dataset shows distinct clustering of species
Petal Features: Petal length and width are highly discriminative features
Setosa Distinction: Setosa species is easily separable from the other classes
High Correlation: Petal measurements show strong correlation
Model Performance: All classifiers achieve high accuracy (>95%) due to well-separated data



##  🗂️ Sales Prediction using Advertising Data

### Project Overview
This project implements machine learning models to predict sales based on advertising expenditure across different media channels (TV, Radio, Newspaper). The goal is to build predictive models that can estimate sales figures given advertising budgets.

### Dataset
#### Source
**advertising.csv** from [Kaggle]( https://www.kaggle.com/datasets/bumba5341/advertisingcsv)
- **Size**: 200 observations × 4 features
- **Features**:
  - TV: Advertising budget spent on TV (in thousands of dollars)
  - Radio: Advertising budget spent on Radio
  - Newspaper: Advertising budget spent on Newspaper
  - Sales: Sales figures (in thousands of units) - **Target Variable**

### Key Features

#### 1. Data Exploration & Analysis
- Descriptive statistics and data overview
- Missing value analysis
- Correlation analysis between features
- Visualization of relationships between advertising channels and sales

#### 2. Visualizations
- Distribution plots of sales
- Scatter plots: TV vs Sales, Radio vs Sales, Newspaper vs Sales
- Correlation heatmap
- Pairplots for feature relationships

#### 3. Machine Learning Models
- **Linear Regression**
- **Random Forest Regressor**
- Model performance evaluation using:
  - Mean Squared Error (MSE)
  - R-squared Score
    
### Prerequisites
- Python 3.7+
- Jupyter Notebook

### Libraries Used
- **pandas** : Data manipulation and analysis
- **numpy**: Numerical computations
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning models and evaluation

### Key Findings
#### Data Insights
- TV advertising shows the strongest correlation with sales
- Radio advertising has moderate correlation with sales
- Newspaper advertising shows weakest correlation
- No missing values or duplicates in the dataset
#### Model Performance
- Linear Regression provides interpretable results
- Random Forest may capture non-linear relationships
- TV spending is the most significant predictor of sales
  
### Results
The project demonstrates how different advertising channels contribute to sales and provides a predictive framework for optimizing advertising budgets.


## 📁 Movie Rating Prediction

### Overview
This project predicts movie ratings using machine learning models trained on the IMDb Movies India dataset. The implementation includes comprehensive data preprocessing, feature engineering, and model evaluation to predict movie ratings based on various features like genre, director, actors, duration, and year of release.


###  Dataset Information
#### Source
**IMDb Movies India.csv** from [Kaggle](https://www.kaggle.com/datasets/harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows)

#### Dataset Features
| Column | Description | Data Type |
|--------|-------------|-----------|
| Name | Movie title | String |
| Year | Release year | String |
| Duration | Movie runtime | String |
| Genre | Movie genre(s) | String |
| Rating | IMDb rating (target variable) | Float |
| Votes | Number of votes | String |
| Director | Movie director | String |
| Actor 1 | Primary actor | String |
| Actor 2 | Secondary actor | String |
| Actor 3 | Supporting actor | String |

####  Dataset Statistics
- **Original Size**: 15,509 movies
- **After Cleaning**: 7,919 movies (removed missing ratings)
- **Time Period**: Various years up to 2021
- **Rating Range**: 1.0 - 10.0

###  Features Implemented
- **Data Preprocessing**: Cleaning and transformation of raw movie data
- **Feature Engineering**:
  - Year, Duration, and Votes extraction
  - Genre multi-label encoding
  - Director and Actors top-N encoding
  - Title length and movie age features
- **Machine Learning Models**:
  - Linear Regression
  - Random Forest Regressor
- **Model Evaluation**: Comprehensive metrics and visualization

###  Results
| Model | MAE | MSE | RMSE | R² |
|-------|-----|-----|------|----|
| Linear Regression | 0.9376 | 1.4381 | 1.1992 | 0.2265 |
| Random Forest | 0.8113 | 1.1704 | 1.0818 | 0.3705 |

**Random Forest outperforms Linear Regression** with better accuracy across all metrics.

### Feature Engineering Details
#### Numeric Features
- **Year_clean**: Extracted from string format
- **Duration_min**: Converted to minutes
- **Votes_clean**: Numeric conversion
- **title_len**: Length of movie title
- **movie_age**: Years since release

#### Categorical Features
- **Genre**: Multi-label binarization (e.g., Action, Drama, Comedy)
- **Director**: Top-50 directors + "Other" category
- **Actors**: Top-80 actors binary encoding

#### Final Feature Matrix
- **Shape**: 7,919 samples × 158 features
- **Train/Test Split**: 80/20 ratio

### Models Implemented
1. Linear Regression
- Baseline model for comparison
- Simple interpretable model

2. Random Forest Regressor
- Ensemble method with 200 estimators
- Better handling of non-linear relationships
- Reduced overfitting through multiple decision trees

### Visualization
- Actual vs Predicted ratings scatter plot
- Residuals distribution analysis
- Model performance comparison

### Key Findings
- Random Forest performs significantly better than Linear Regression
- Feature engineering greatly improves model performance
- Actor and director information are strong predictors of movie ratings
- Movie age has moderate correlation with ratings

### Future Improvements
- Hyperparameter tuning with GridSearchCV
- Additional features (budget, box office collection)
- Advanced models (XGBoost, Neural Networks)
- Sentiment analysis of movie reviews
- Web application for real-time predictions
- Cross-validation for more robust evaluation
- Handle class imbalance in ratings

## License
This project is open source and available under the MIT License.

## Author
Hiba Ali

GitHub: @its89ba

## ⭐ If you find this project helpful, please give it a star!
