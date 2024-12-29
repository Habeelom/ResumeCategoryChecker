# Resume Screening App

## Overview
This web application uses machine learning to categorize resumes into various job roles. It leverages a combination of natural language processing (NLP) techniques and a machine learning model to predict the category of a given resume.

## How It's Made
**Tech Used:** Python, Streamlit, Scikit-learn, NLTK, Matplotlib, Seaborn, Jupyter Notebook

This project demonstrates the following key features:
- **Data Cleaning:** Utilizes regular expressions for thorough text cleaning.
- **NLP Techniques:** Leverages TF-IDF Vectorizer for text feature extraction.
- **Machine Learning Model:** This model implements a K-Nearest Neighbors (KNN) classifier wrapped in a OneVsRestClassifier for multi-class classification.
- **Visualization:** Uses Seaborn and Matplotlib for visual data exploration.

### Key Components
1. **Data Loading and Exploration:**
    - Load resume data from a CSV file.
    - Explore dataset characteristics such as category distribution.

2. **Data Cleaning:**
    - Implement a custom function `clean_resume` to remove URLs, special characters, and extra spaces from resume text.

3. **Label Encoding:**
    - Convert categorical labels into numeric format using `LabelEncoder`.

4. **TF-IDF Vectorization:**
    - Transform resume text into numerical feature vectors using `TfidfVectorizer`.

5. **Model Training:**
    - Split the data into training and test sets.
    - Train a KNN classifier with the training data.
    - Evaluate the model's accuracy on the test set.

6. **Pickle Serialization:**
    - Save the trained TF-IDF vectorizer and KNN classifier for later use.

7. **Streamlit App:**
    - Create an intuitive interface for users to upload resumes and view predictions.
    - Display the predicted job category for the uploaded resume.

## Optimizations
- **Efficient Data Cleaning:** Improved resume text cleaning with regular expressions.
- **Model Performance:** Achieved high accuracy with optimized KNN classifier.
- **User Interface:** Enhanced user experience with a clean and responsive Streamlit interface.

## Lessons Learned
- **NLP Techniques:** Gained hands-on experience with text cleaning and feature extraction.
- **Model Training:** Improved understanding of multi-class classification with scikit-learn.
- **Streamlit Development:** Learned to build interactive web applications with Streamlit.
