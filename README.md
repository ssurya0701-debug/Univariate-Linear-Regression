# Implementation of Univariate Linear Regression
## Aim:
To implement univariate Linear Regression to fit a straight line using least squares.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
1.	Get the independent variable X and dependent variable Y.
2.	Calculate the mean of the X -values and the mean of the Y -values.
3.	Find the slope m of the line of best fit using the formula.
 ![eqn1](./eq1.jpg)
4.	Compute the y -intercept of the line by using the formula:
![eqn2](./eq2.jpg)  
5.	Use the slope m and the y -intercept to form the equation of the line.
6.	Obtain the straight line equation Y=mX+b and plot the scatterplot.
## Program
```
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

# Load the dataset
data = datasets.load_diabetes()
X = data.data
y = data.target

# Split training and testing data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=1
)

# Create linear regression object
reg = linear_model.LinearRegression()

# Train the model
reg.fit(X_train, y_train)

# Regression coefficients
print("Coefficients:", reg.coef_)

# Variance score
print("Variance score: {}".format(reg.score(X_test, y_test)))

# Plot residual errors
plt.style.use('fivethirtyeight')

# Training data residuals
plt.scatter(
    reg.predict(X_train),
    reg.predict(X_train) - y_train,
    color="green", s=10, label="Train data"
)

# Testing data residuals
plt.scatter(
    reg.predict(X_test),
    reg.predict(X_test) - y_test,
    color="blue", s=10, label="Test data"
)

# Zero residual error line
plt.hlines(y=0, xmin=0, xmax=500, linewidth=2)

# Legend and title
plt.legend(loc="upper right")
plt.title("Residual errors")
plt.xlabel("Predicted values")
plt.ylabel("Residuals")

plt.show()





```
## Output
<img width="837" height="611" alt="Screenshot 2026-11-25 103247" src="https://github.com/user-attachments/assets/11a6be2a-e0cf-4602-84a1-c7b4b3a20e16" />


## Result
Thus the univariate Linear Regression was implemented to fit a straight line using least squares.
