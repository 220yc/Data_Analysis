# -House-Price-Prediction🏠🏠🏠🔍
House price prediction enables better decision-making, risk assessment, and market analysis in the context of housing investments and economic trends.

This goal is to accurately predict house prices to identify potential opportunities for profitable investments, financial planning and making informed decisions regarding housing investments.

# Getting Started
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
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/f1b8c91a-20db-48fc-91dd-0aa83789acc1" width="200" height="180"/>
<div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/b7c53b08-1d1b-4f2d-b073-027b19bf4ea0)" width="400" height="300"/></div>

**Convert a skewed distribution into a more symmetric, approximately normal distribution.**
```
df['單價'] = np.log1p(df['單價'])
sns.distplot(df['單價'])
plt.show()
```
<div align=center><div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/1ce271f9-0a84-449c-b882-17b878028c99" width="400" height="300"/></div>

**The Q-Q plot is used to visually check if the data follows a normal distribution.**
```
from scipy import stats
res = stats.probplot(df['單價'], plot=plt)
```
<div align=center><div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/76d67523-1ebf-4e2c-ae08-e66b6f04bcb8" width="400" height="300"/></div>

This states that the presence of heavy tails in the plot indicates a higher probability of extreme outliers. This can potentially affect the accuracy of predictions or statistical analyses.

**Correlation Matrix**
```
corrmat= df.corr()
# corrmat['單價'].abs().sort_values(ascending=False)

k=13
corrmat[corrmat['單價'].abs() > 0.5]['單價']
cols = corrmat['單價'].abs().nlargest(k)
cols=cols.index.to_list()
cm = df[cols].corr()
plt.figure(figsize=(10,8))
sns.heatmap(cm,annot=True,square=True)
plt.show()
```
<div align=center><div align=center><img src="https://github.com/220yc/-House-Price-Prediction/assets/91858697/b98fed15-8f28-47a4-90f9-696474920d98" width="400" height="300"/></div>











