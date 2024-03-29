# Classification of Arrhythmia
Data Science Project

## Introduction

This project aims to develop a machine learning model that can detect the presence of arrhythmia in electrocardiogram (ECG) recordings. 

## Project Organization

├── README.md          <- The top-level README for developers using this project.

├── notebooks          <- This folder contains all scripts as jupyter notebooks needed for data analysis and preprocessing, training and testing of models and model interpretation analysis and corresponding visualizations for both the UCI-Bilkent dataset (1.A_a and 1.B_a) and the MIT-BIH dataset (2.A_a, 2.B_a and 2.C_a) respectively. During the feature engineering and machine learning and deep learning paart, different options for feature engineering and model set up were explored. Alternatives for the UCI-Bilkent modeling (Alternative_1B) and for the MIT-BIH dataset modelling and deep learning part (Alternative_2.B, Alternative_2.C) can also be found in the same directory. 

├── reports            <- This folder contains the final project report including all figures in a separate subfolder. 

## Datasets

Two different datasets were used in this project - UCI-Bilkent dataset and the MIT-BIH databaset. 
The “Cardiac Arrhythmia Database donated to UCI by Bilkent” contains electrocardiogram (ECG) information that can be used for diagnosis of arrhythmia [1]. 
It consists of 452 entries (rows) and 280 columns, containing 279 attributes and one target variable (class).  
The MIT-BIH Arrhythmia Database consists of 48 half-hour segments of two-channel ambulatory ECG recordings obtained from 47 subjects during the years 1975 to 1979 by the BIH Arrhythmia Laboratory [3]. The dataset is freely available on PhysioNet [2].
The datset used in this project [4] consists of 100689 samples and 34 columns, with 33 features pertinent to ECG analysis and one target column denoting arrhythmia classifications.

## Methods

**Exploratory Data Analysis (EDA):**
EDA was performed to understand the characteristics and patterns of the data, specifically to examine the distribution of data, outliers, and missing values.
These steps were performed for each dataset respectively. 
The corresponding notebooks are labelled 1A-EDA-UCI-Bilkent-dataset and 2A-EDA-MIT-BIH-dataset.

**Data Preprocessing and Feature Engineering:**
Principal Component Analysis (PCA) was applied to the UCI-Bilkent dataset to reduce the number of features while preserving the most important ones present in the data. The preprocessed dataset used for the subsequent modelling steps contained 78 features that explain more than 90% of the variance in the data. 
For the MIT-BIH dataset, we strategically downsampled our data Ito address the significant class imbalance to the minority class of abnormal heartbeat, resulting in an evenly balanced dataset with a total of 21212 samples.

**Model Building:**
A train-test split was performed to evaluate the performance of the models, and feature scaling was applied to normalize the data. Different machine learning models were trained and tested on the preprocessed data sets. GridSearchCV (UCI-Bilkent dataset) and Randomized Search (MIT-BIH dataset) were performed for hyperparameter tuning. 
The results of these steps were analyzed with multiple visualizations to select the best-performing model.
These steps were performed for both datasets respectively. 
The corresponding notebooks are labelled 1B-Modelling-UCI-Bilkent-dataset and 2B-Modelling-MIT-BIH-dataset.

**Deep-Learning**
Different architectures of Dense Neural Networks and Artificial Neural Networks were explored for the MIT-BIH dataset. For this part, we conducted an additional downsampling procedure to optimize computational efficiency while retaining sufficient data points to a total sample size of 10000.
The corresponding notebook is labelled 2C-DeepLearning-MIT-BIH-dataset.

## Results

Based on our results, minimizing false negatives is crucial for our project's success. Gradient Boosting achieved the best performance among the models evaluated concerning the number of false negatives, with a very high accuracy of 98%. Additionally, Random Forest exhibited strong performance with a 97% accuracy, while AdaBoosting and XGBoosting achieved accuracies of 98%. Comparing deep learning models to traditional machine learning algorithms, DNN and ANN models achieved respectable accuracies ranging from 95% to 96%.
The strong performance of models such as Gradient Boosting, XGBoost, and Random Forest, combined with their ability to effectively minimize false negatives, positions them as invaluable tools for aiding healthcare professionals in diagnosing arrhythmias. These findings underscore the considerable potential for deployment in clinical environments like hospitals and healthcare facilities.

## Contributors

- James
- Miran
- Theresa 

## References
[1] Guvenir, H. Altay, Burak Acar, Gulsen Demiroz, Ayhan Cekin. (1997). A Supervised Machine Learning Algorithm for Arrhythmia Analysis. In Proceedings of the Computers in Cardiology Conference, Lund, Sweden.
[2] Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online], 101(23), pp. e215–e220.
[3] Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. IEEE Engineering in Medicine and Biology Magazine, 20(3), 45-50.
[4] Sakib, S., Fouda, M. M., & Fadlullah, Z. M. (2021). Harnessing Artificial Intelligence for Secure ECG Analytics at the Edge for Cardiac Arrhythmia Classification. In Secure Edge Computing (1st ed., pp. 17). CRC Press. ISBN: 9781003028635.

