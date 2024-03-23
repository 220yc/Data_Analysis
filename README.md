# -House-Price-Prediction🏠🏠🏠🔍
House price prediction enables better decision-making, risk assessment, and market analysis in the context of housing investments and economic trends.

This goal is to accurately predict house prices to identify potential opportunities for profitable investments, financial planning and making informed decisions regarding housing investments.

# Getting Started
Import the library
```
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn import datasets as ds
import numpy as np
```
Load the dataset
```
df = pd.read_csv("training_data.csv",encoding='utf-8')
df
```

```
df.info()
df.describe()
```
These lines of code display information about the DataFrame, including column names, data types, and the number of non-null values. The `df.describe()` function provides statistical summary of the numerical columns in the DataFrame.

```
df.drop('ID', axis=1, inplace=True)
df = df.drop(['remarks'],axis=1)
df = df.drop(['usage zone'],axis=1)
df
```
1. 'ID': This column is being dropped as it is deemed unnecessary and does not provide any meaningful information for analysis.
2. 'remarks': This column is being dropped because it has a significant number of missing values (only 92 non-null values out of the total number of rows). Since the missing values are substantial, it is decided to remove this column altogether.
3. 'usage zone': This column is being dropped because it appears that the majority of values in this column are 'None'. If the 'None' values dominate the column, it may not be useful for analysis and hence is removed.













