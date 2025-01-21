# Diamond Clustering and Analysis

This project applies clustering techniques to analyze a dataset of diamonds, focusing on segmentation by price, weight, and quality characteristics. The goal is to identify premium, mid-range, and budget segments in the diamond market while providing insights into key features like cut, color, and clarity.

# Features

Data Cleaning and Preprocessing:

Handles missing values using the most frequent value imputation.
Normalizes numerical columns using StandardScaler.
Encodes categorical variables using OneHotEncoder.

Clustering with K-Means:

Groups diamonds into three clusters based on combined numerical and categorical features.
Segments the market into premium, mid-range, and budget categories.

Anomaly Detection:

Uses IsolationForest to identify outliers in the dataset.

Cluster Analysis:

Calculates cluster summaries, including average price, weight, and other attributes.
Visualizes clusters based on weight and price.

Qualitative Analysis:

Analyzes categorical features (Cut, Color, Clarity) across clusters.
Visualizes the relationship between weight, price, and qualitative attributes.
