## Spam Email Classification Project
Overview
This project aims to build a robust spam email classifier using machine learning techniques. It involves several steps, from data exploration and feature engineering to model training, hyperparameter tuning, and deployment using Streamlit.

Dataset
Source: https://www.kaggle.com/datasets/ashfakyeafi/spam-email-classification

Description: This dataset contains a collection of emails labeled as either "spam" or "ham" (not spam). It serves as the foundation for training and evaluating our spam detection model.

Methodology
1. Exploratory Data Analysis (EDA)
Conducted comprehensive EDA to understand data characteristics, distributions, and patterns.
Identified key features and potential challenges in the dataset.

2. Feature Engineering
Extracted relevant features from the email text to improve model performance.
Leveraged the NLTK library for text processing, including tokenization, stemming, and removal of stop words and punctuation.

3. Text Vectorization
Utilized CountVectorizer and TfidfVectorizer to transform text data into numerical vectors suitable for machine learning models.
TfidfVectorizer was chosen for its ability to weigh words based on their importance in the context of the entire corpus, enhancing model accuracy.

4. Model Training and Evaluation
Trained and evaluated several classification models, including:

Logistic Regression
Support Vector Machines (SVM)
Naive Bayes
Random Forest
DecisionTreeClassifier
GradientBoostingClassifier
AdaBoostClassifier
XGBClassifier

Models were organized in a dictionary for streamlined training and testing.
Custom functions were created to train and evaluate each model, providing flexibility and ease of use.

5. Hyperparameter Tuning
Fine-tuned model hyperparameters using techniques like GridSearchCV to maximize performance.
Focused on optimizing for both precision and accuracy to minimize false positives and false negatives.

6. Model Selection and Persistence
Selected the best performing model based on evaluation metrics.
Saved the trained model and the TfidfVectorizer as pickle files for easy deployment and future use.

Technologies Used:
Python
Pandas
Numpy
Matplotlib
Scikit-learn
NLTK
Streamlit
Pickle

Deployment:
Created a user friendly interface using Streamlit for real-time spam detection.
The app allows users to input email text and receive a prediction of whether the email is spam or not.