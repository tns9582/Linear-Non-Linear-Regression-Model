## Project Overview
This project demonstrates a complete regression modeling pipeline, combining both traditional statistical approaches and modern machine learning practices to analyze synthetic data. It begins with linear regression and evolves into nonlinear modeling using sinusoidal features and gradient-based optimization techniques in both NumPy and TensorFlow.

## Tools & Technology
  * Languages: Python (NumPy, Pandas, Matplotlib, Seaborn, TensorFlow)
  * Exploratory Data Analysis (EDA): Scatterplots, correlation computation, and trend identification.
  * Linear Regression: Implemented from scratch using NumPy, including manual loss computation and parameter optimization.
  * Loss Functions: Mean Squared Error (MSE) for evaluating model performance.
  * Model Optimization:
      * Closed-form solution for linear regression.
      * Gradient Descent (manual and automatic differentiation via TensorFlow).
  * Nonlinear Modeling: Extended the linear model by adding a cosine feature to capture periodic trends in residuals.
  * TensorFlow for Custom ML Models: Reimplementation of models with tf.Variable, tf.GradientTape, and training loops.

## Key Objective
1. Data Exploration & Linear Modeling
  * Loaded a synthetic dataset with two continuous variables x and y.
  * Visualized the data with scatterplots.
  * Observed a linear relationship with sinusoidal residuals.
  * Built a linear regression model using:
      * Custom model: 𝑦^= 𝜃⋅𝑥
      * MSE loss function
      * Closed-form solution to find optimal 𝜃

2️. Residual Analysis
  * Visualized residuals from the linear model.
  * Identified sinusoidal patterns in residuals ➡ indicated underfitting.
    
3️. Nonlinear Regression with Sinusoidal Feature
  * Enhanced the model to: 𝑦^ = 𝜃1⋅𝑥 + cos(𝜃2⋅𝑥)
  * Defined custom derivatives for both parameters.
  * Implemented gradient descent manually to minimize loss.

4️. TensorFlow Implementation
  * Re-implemented the nonlinear model using TensorFlow:
      * Defined model using tf.math.cos
      * Used tf.GradientTape for automatic differentiation
      * Ran custom training loops
  * Compared convergence with NumPy version.

## Visualizations
  * Scatter plots of actual vs predicted values.
  * Loss curves over varying parameter values.
  * Residual plots for linear vs nonlinear models.
  * Model evolution over gradient descent iterations.

## Conclusion
This project demonstrates the progression from a basic linear regression model to a more advanced nonlinear approach by incorporating sinusoidal features to better fit the data. Through careful residual analysis and the implementation of custom gradient descent optimization, the model's performance was significantly enhanced. Additionally, re-creating the model using TensorFlow showcased the practical application of automatic differentiation and custom training loops.
