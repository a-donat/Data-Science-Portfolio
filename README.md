# Data-Science-Portfolio
## Classification: Breast Cancer Detection
This project highlights the following:

      1. My mastery of the major classification methods in machine learning.
      2. My ability to use finetune the hyperparameters of these analyses. 
      3. My ability to construct pipelines that connect data processing with data analysis.
      4. My understanding of the advantages and disadvantages of each method, 
      and how to combine methodologies accordingly.
      5. My superior data visualization skills.

Libraries Used: numpy, pandas, sklearn, matplotlib, seaborn

Outline:

    1. Set-Up
    2. Data Exploration
    3. Analysis: Model Evaluation and Hyperparameter Tuning
        A. Principal Component Analysis
        B. Decision Tree
        C. Gaussian Naive Bayes
        D. K Nearest Neighbors
        E. Logistic Regression
        F. Support Vector Machine
    4. Model Comparison
        A. Learning Curves
              ...
        E. Performance Metrics
            - Accuracy, Recall, Precision, and F1 Scores
    5. Ensemble Predictions
        A. Relationship Between Errors
        B. Implement Voting on Training Set
    6. Performance on Test Set

# An Interactive Explanation of the Principal Component Analysis (PCA)

Outline : 

      1. Set-Up
      2. Dataset
      3. Description of the Principal Component Analysis
      4. PCA Steps
            A. Standardize the data
            B. Construct the covariance matrix
                  a. What is covariance?
                  b. Visualizing Covariance
            C. Obtain the eigenvalues and eigenvectors of the covariance matrix
                  a. What are eigenvalues and eigenvectors?
                  b. Visualizing Eigenvectors and Eigenvalues
            D. Create matrix of ranked eigenvectors
            E. Multiply Data by Eigenvector Matrix
                  a. How does multiplication by the Eigenvector Matrix transform the data?
            F. Reduce Dimensionality of PCA Transformed Dataset
      5. PCA in SK-Learn
      6. Disadvantages and Limits of PCA
