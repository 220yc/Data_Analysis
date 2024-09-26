
House price prediction enables better decision-making, risk assessment, and market analysis in the context of housing investments and economic trends.

The goal is to accurately predict house prices to identify potential opportunities for profitable investments, financial planning and making informed decisions regarding housing investments.

# Data Analysis
**Import the library**
```
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets as ds
import numpy as np
```
**Load the dataset**
```
df = pd.read_csv("training_data.csv",encoding='utf-8')
df
```
**Inspect the types of feature columns**
```
df.info()
df.describe()
```
These lines of code display information about the DataFrame, including column names, data types, and the number of non-null values. The `df.describe()` function provides statistical summary of the numerical columns in the DataFrame.  
  
**Drop the ID column as it is not necessary for model training**  
```
df.drop('ID', axis=1, inplace=True)
df = df.drop(['remarks'],axis=1)
df = df.drop(['usage zone'],axis=1)
df
```
1. 'ID': This column is being dropped as it is deemed unnecessary and does not provide any meaningful information for analysis.
2. 'remarks': This column is being dropped because it has a significant number of missing values (only 92 non-null values out of the total number of rows). Since the missing values are substantial, it is decided to remove this column altogether.
3. 'usage zone': This column is being dropped because it appears that the majority of values in this column are 'None'. If the 'None' values dominate the column, it may not be useful for analysis and hence is removed.   
  
**Statistical summary & Creates a distribution plot**  
This code prints the statistical summary of the '單價' column in the DataFrame. The `describe()` function provides information such as count, mean, standard deviation, minimum value, maximum value, and quartiles.  
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/f1b8c91a-20db-48fc-91dd-0aa83789acc1" width="230" height="180"/></div>
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/b7c53b08-1d1b-4f2d-b073-027b19bf4ea0)" width="400" height="300"/></div>

**Convert a skewed distribution into a more symmetric, approximately normal distribution.**
```
df['單價'] = np.log1p(df['單價'])
sns.distplot(df['單價'])
plt.show()
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/1ce271f9-0a84-449c-b882-17b878028c99" width="400" height="300"/></div>

**The Q-Q plot is used to visually check if the data follows a normal distribution.**
```
from scipy import stats
res = stats.probplot(df['單價'], plot=plt)
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/76d67523-1ebf-4e2c-ae08-e66b6f04bcb8" width="400" height="300"/></div>

This states that the presence of heavy tails in the plot indicates a higher probability of extreme outliers. This can potentially affect the accuracy of predictions or statistical analyses.
  
# Features
**Correlation Matrix**
```
corrmat= df.corr()
#corrmat['單價'].abs().sort_values(ascending=False)

k=13
corrmat[corrmat['單價'].abs() > 0.5]['單價']
cols = corrmat['單價'].abs().nlargest(k)
cols=cols.index.to_list()
cm = df[cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,square=True)
plt.show()
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/b98fed15-8f28-47a4-90f9-696474920d98" width="400" height="300"/></div>
The correlation matrix shows the pairwise correlation coefficients between all pairs of columns.
  
**Separate numerical features and categorical features**
```
# Initialize empty lists to store the names of numerical features and categorical features.
num_features = []
cate_features = []

for col in df.columns:
    if df[col].dtype == 'object':
        cate_features.append(col)
    else:
        num_features.append(col)
print('number of numeric features:', len(num_features))
print('number of categorical features:', len(cate_features))
```
<img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/686174b2-d7c4-4862-9c76-b4a8595bba89" width="250" height="50"/>
  
**Relationship between the numerical features and '單價'**
```
plt.figure(figsize=(16, 20))
plt.subplots_adjust(hspace=0.3, wspace=0.3)
for i, feature in enumerate(num_features):
    plt.subplot(9, 4, i+1)
    sns.scatterplot(x=feature, y='單價', data=df, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel('單價')
plt.show()
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/10feeea7-bcbc-4a52-bdf1-e7b9d26ed639" width="800" height="600"/></div>

**Relationship between '縣市' and  '單價'**
```
plt.figure(figsize=(8, 6))
sns.boxplot(x='縣市', y='單價', data=df)
plt.xlabel('縣市', fontsize=14)
plt.ylabel('單價', fontsize=14)
plt.xticks(rotation=90, fontsize=12)
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/b36901f1-8b65-455b-b953-97cc2ceff69b" width="500" height="300"/></div>  

```
# store the unit price data for each city.
city_prices = {}

# This loop iterates through each unique city
for city in df['縣市'].unique():
    city_prices[city] = df[df['縣市'] == city]['單價']

# displays by histograms
plt.figure(figsize=(12, 8))
for city, prices in city_prices.items():
    sns.histplot(prices, bins=20, kde=True, label=city)
plt.xlabel('單價')
plt.ylabel('頻率')
plt.legend()
plt.title('各縣市的單價分布')
plt.show()
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/876587a7-fc87-4ed2-953c-a012a3dc9d9f" width="500" height="300"/></div>  

**Calculate the average price for each area '縣市', '地區'**
It then iterates through each county and creates a bar chart to compare the average prices across different areas within that county.  
```
# Calculate average price for each area
average_prices_by_area = df.groupby(['縣市', '地區'])['單價'].mean().reset_index()
 
counties = average_prices_by_area['縣市'].unique()

for county in counties:
    selected_data = average_prices_by_area[average_prices_by_area['縣市'] == county]
    plt.figure(figsize=(6, 4))
    plt.barh(selected_data['地區'], selected_data['單價'], color='skyblue')
    plt.xlabel('平均價格', fontsize=14)
    plt.ylabel('地區', fontsize=14)
    plt.title(f'{county}內不同地區的平均價格比較', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/f9b856b2-e7af-479e-90e4-9fcfe13478cf" width="500" height="300"/></div>  
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/1695d5c9-2640-44e8-b482-78dac6404ee7" width="500" height="300"/></div>  
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/54bc9545-e20b-4bd2-a421-1dcfee03a18c" width="500" height="300"/></div>  
  
**Calculate the average price for each area '建物型態','主要用途'**
```
average_prices_by_area = df.groupby(['建物型態','主要用途'])['單價'].mean().reset_index()

counties = average_prices_by_area['建物型態'].unique()

for county in counties:
    selected_data = average_prices_by_area[average_prices_by_area['建物型態'] == county]
    plt.figure(figsize=(6, 4))
    plt.barh(selected_data['主要用途'], selected_data['單價'], color='skyblue')
    plt.xlabel('平均價格', fontsize=14)
    plt.ylabel('主要用途', fontsize=14)
    plt.title(f'{county}內不同主要用途的平均價格比較', fontsize=16)
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()
```
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/a7184903-51d4-41a6-9595-e3225a430d96" width="400" height="600"/></div>  


**Perform K-means clustering on the average prices by area for each county.**
It assigns a '地區類別' (area category) to each data point based on the clustering results. 
```
from sklearn.cluster import KMeans
# Step 1: Calculate the average price by area for each county
average_prices_by_area = df.groupby(['縣市', '地區'])['單價'].mean().reset_index()
counties = average_prices_by_area['縣市'].unique()

clustered_data = []
for county in counties:
    county_data = df[df['縣市'] == county].copy()

    # Step 2: Perform area clustering on each county's data subset
    area_prices = county_data['單價'].values.reshape(-1, 1)

    n_clusters = 4  # Assuming 2 clusters for dividing areas
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(area_prices)
    county_data['地區類別'] = kmeans.labels_

    clustered_data.append(county_data)

# Step 3: Merge the clustered data back to the original dataset
df_clustered = pd.concat(clustered_data)

# Print the result
df['地區類別'] = df_clustered['地區類別']
df
```
**Perform One-Hot Encoding on feature columns**
```
df_encoded = pd.get_dummies(df, columns=['縣市','主要用途', '主要建材','建物型態'])  
```
# Feature selection
Feature selection is an important step in machine learning to identify the most relevant features that contribute the most to the prediction task while reducing complexity and overfitting. 
```
from sklearn.decomposition import PCA

# Get the names of all features
features = list(X.columns)

# Create and fit a PCA model
pca = PCA(n_components=2)
pca.fit(X)

# Get the feature weights of the principal components
components = pca.components_

# Display the feature weights of the principal components and their corresponding feature names
for i, component in enumerate(components):
    print(f"Principal Component {i + 1}:")
    for j, weight in enumerate(component):
        feature_name = features[j]
        print(f"{feature_name}: {weight}")
```

**Random Forest**
```
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
y = df_encoded['單價']
X = df_encoded.drop(['單價'],axis=1)
X
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,random_state=24)
rfModel = RandomForestRegressor(n_estimators=400,random_state=24)
rfModel.fit(X_train, y_train)

y_pred =rfModel.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('Random Forest R square:', r2)
```
- Random Forest R square: 0.9132248925945137  

**Random Forest Regressor**
```
from sklearn.model_selection import cross_val_score
rfModel_cv = RandomForestRegressor(n_estimators=100, random_state=42)
scores = cross_val_score(rfModel_cv, X, y, cv=5, scoring='r2')
print("Random Forest Regressor R square: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
```
- Random Forest Regressor R square: 0.90 (+/- 0.05)  

**Linear Regression**
```
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Split the data into feature matrix (X) and target variable (y)
y = df_encoded['單價']
X = df_encoded.drop(['單價'], axis=1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
linear_model = LinearRegression()

# Fit the model using the training data
linear_model.fit(X_train, y_train)

# Predict
y_pred = linear_model.predict(X_test)

# Evaluate the performance
r2 = r2_score(y_test, y_pred)
print('Linear Regression R square:', r2)
```
- Linear Regression R square: 0.6907120578357775  

**K-Nearest Neighbors (KNN)**
```
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

# Create a KNN regression model
knn_model = KNeighborsRegressor(n_neighbors=5)

# Fit the model using the training data
knn_model.fit(X_train, y_train)

# Predict
y_pred = knn_model.predict(X_test)

# Evaluate the performance
r2 = r2_score(y_test, y_pred)
print('KNN R square:', r2)
```
- KNN R square: 0.78788082885659  

**Decision Tree Regression**
```
from sklearn.tree import DecisionTreeRegressor

# Load the data and prepare X and y
X = df_encoded.drop(['單價'], axis=1)
y = df_encoded['單價']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a decision tree regression model
dt_model = DecisionTreeRegressor(random_state=42)

# Train the model
dt_model.fit(X_train, y_train)

# Predict
y_pred = dt_model.predict(X_test)

# Evaluate the performance
r2 = r2_score(y_test, y_pred)
print('Decision Tree R square:', r2)
```
- Decision Tree R square: 0.8007312652631591  

**Gradient Boosting Regression**
```
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Create a Gradient Boosting regression model
gb_model = GradientBoostingRegressor(n_estimators=200, random_state=60)

# Fit the model using the training data
gb_model.fit(X_train, y_train)

# Predict
y_pred = gb_model.predict(X_test)

# Evaluate the performance
r2 = r2_score(y_test, y_pred)
print('Gradient Boosting R square:', r2)
```
- Gradient Boosting R square: 0.9020817312998458
<img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/87549cc6-d62c-4a7f-91bb-6d1d592c11e5" width="150" height="250"/>  
  
# Data Analysis
**Test Result**
