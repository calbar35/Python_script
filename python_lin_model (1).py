#!/usr/bin/env python

# ## Python Linear Regresssion Assingment
# ## BSGP 7020: Callie Barber

# ### Import pandas, sklearn and matplotlib
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import sys
import os 

print("Running linear modeling of data python script/n")
print()

# ### Set notebook variables
filename = sys.argv[1]
base,ext = os.path.splitext(filename)
print(base)
print(ext)

print("loading filename {}".format(filename))
print()

# ### Use read_csv() to read regrex1.csv file 
dataset = pd.read_csv(filename)
dataset.describe()
dataset

# ### Fitting Linear Regression to the Dataset
model = LinearRegression()
model.fit(dataset[['x']], dataset[['y']])
LinearRegression
LinearRegression()

# ### Visualizing the Linear Regression results
# ### Scatter plot of original dataset
# ### Linear model of dataset
plt.title('y vs x')
plt.title('Linear model of y vs x')
plt.xlabel('x')
plt.ylabel('y')

# ### Scatter plot of original dataset
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.title("y vs x for {}".format(base))
plt.xlabel('x')
plt.ylabel('y')         
plt.savefig("{}.png".format(base))

# ### Combined plot
plt.scatter(dataset[['x']], dataset[['y']], color = 'red')
plt.plot(dataset[['x']], model.predict(dataset[['x']]), color = 'blue')
plt.title("model of y vs x for {}".format(base))
plt.xlabel('x')
plt.ylabel('y')
plt.savefig("{}_model.png".format(base))