```python
# Importing Requierd Libraries
import pandas as pd
import matplotlib.pyplot as plt
#plt.style.use
import seaborn as sns
import numpy as np
%matplotlib inline
from jupyterthemes import jtplot

jtplot.style(theme='onedork')
jtplot.style(context='talk', fscale=1.4, spines=False, gridlines='--')
jtplot.style(ticks=True, grid=False, figsize=(6, 4.5))

#removing warnings
import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
# sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.cluster import Birch
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans 
from sklearn import preprocessing
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import sklearn.cluster as cluster
import time
```


```python
#import data with \ as delimiter
data = pd.read_csv('D:\School\PythonDataScience\marketing_campaign.csv', delimiter='\t')
data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Year_Birth</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>Dt_Customer</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>NumWebVisitsMonth</th>
      <th>AcceptedCmp3</th>
      <th>AcceptedCmp4</th>
      <th>AcceptedCmp5</th>
      <th>AcceptedCmp1</th>
      <th>AcceptedCmp2</th>
      <th>Complain</th>
      <th>Z_CostContact</th>
      <th>Z_Revenue</th>
      <th>Response</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>1957</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>04-09-2012</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>1954</td>
      <td>Graduation</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>08-03-2014</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>1965</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>21-08-2013</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>1984</td>
      <td>Graduation</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>10-02-2014</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>1981</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>19-01-2014</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>10870</td>
      <td>1967</td>
      <td>Graduation</td>
      <td>Married</td>
      <td>61223.0</td>
      <td>0</td>
      <td>1</td>
      <td>13-06-2013</td>
      <td>46</td>
      <td>709</td>
      <td>...</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>4001</td>
      <td>1946</td>
      <td>PhD</td>
      <td>Together</td>
      <td>64014.0</td>
      <td>2</td>
      <td>1</td>
      <td>10-06-2014</td>
      <td>56</td>
      <td>406</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>7270</td>
      <td>1981</td>
      <td>Graduation</td>
      <td>Divorced</td>
      <td>56981.0</td>
      <td>0</td>
      <td>0</td>
      <td>25-01-2014</td>
      <td>91</td>
      <td>908</td>
      <td>...</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>8235</td>
      <td>1956</td>
      <td>Master</td>
      <td>Together</td>
      <td>69245.0</td>
      <td>0</td>
      <td>1</td>
      <td>24-01-2014</td>
      <td>8</td>
      <td>428</td>
      <td>...</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>9405</td>
      <td>1954</td>
      <td>PhD</td>
      <td>Married</td>
      <td>52869.0</td>
      <td>1</td>
      <td>1</td>
      <td>15-10-2012</td>
      <td>40</td>
      <td>84</td>
      <td>...</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>11</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2240 rows × 29 columns</p>
</div>




```python
tfont = {'fontname':'Monsterrat'}
nfont = {'fontname':'Pacifico'}
```


```python
#DATA DESCRIPTION/WRANGLING
```


```python
# possibly combine total amount of products and total purchases?
data['mnt_products'] = data['MntWines'] + data['MntFruits'] + data['MntMeatProducts'] + data['MntFishProducts'] +\
                        data['MntSweetProducts'] + data['MntGoldProds']
```


```python
# Categorize columns for self reference 
#all numeric columns = ['income', 'kidhome', 'teenhome', 'recency', 'mntwines', 'mntfruits', 'mntmeatproducts', 'mntfishproducts', 'mntsweetproducts', 
                        #'mntgoldprods', 'numdealspurchases', 'numwebpurchases',  'numcatalogpurchases', 'numstorepurchases', 'numwebvisitsmonth',  'acceptedcmp3', 
                        #'acceptedcmp4', 'acceptedcmp5', 'acceptedcmp1', 'acceptedcmp2', 'complain', 'z_costcontact', 'z_revenue', 'response']

#numeric columns most likely correlating = ['Income', 'Kidhome','Teenhome', 'Recency', 'MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
                    #'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']

#boolean columns = ['acceptedcmp3', 'acceptedcmp4', 'acceptedcmp5', 'acceptedcmp1','acceptedcmp2', 'complain', 'response']

#categorical columns = ['education', 'marital_status']
#date columns = ['year_birth','dt_customer']
```


```python
#to see if the data is in the right format
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 30 columns):
     #   Column               Non-Null Count  Dtype  
    ---  ------               --------------  -----  
     0   ID                   2240 non-null   int64  
     1   Year_Birth           2240 non-null   int64  
     2   Education            2240 non-null   object 
     3   Marital_Status       2240 non-null   object 
     4   Income               2216 non-null   float64
     5   Kidhome              2240 non-null   int64  
     6   Teenhome             2240 non-null   int64  
     7   Dt_Customer          2240 non-null   object 
     8   Recency              2240 non-null   int64  
     9   MntWines             2240 non-null   int64  
     10  MntFruits            2240 non-null   int64  
     11  MntMeatProducts      2240 non-null   int64  
     12  MntFishProducts      2240 non-null   int64  
     13  MntSweetProducts     2240 non-null   int64  
     14  MntGoldProds         2240 non-null   int64  
     15  NumDealsPurchases    2240 non-null   int64  
     16  NumWebPurchases      2240 non-null   int64  
     17  NumCatalogPurchases  2240 non-null   int64  
     18  NumStorePurchases    2240 non-null   int64  
     19  NumWebVisitsMonth    2240 non-null   int64  
     20  AcceptedCmp3         2240 non-null   int64  
     21  AcceptedCmp4         2240 non-null   int64  
     22  AcceptedCmp5         2240 non-null   int64  
     23  AcceptedCmp1         2240 non-null   int64  
     24  AcceptedCmp2         2240 non-null   int64  
     25  Complain             2240 non-null   int64  
     26  Z_CostContact        2240 non-null   int64  
     27  Z_Revenue            2240 non-null   int64  
     28  Response             2240 non-null   int64  
     29  mnt_products         2240 non-null   int64  
    dtypes: float64(1), int64(26), object(3)
    memory usage: 525.1+ KB
    


```python
# need to change date incorrectly formatted data and format to liking
data['Dt_Customer'] = pd.to_datetime(data['Dt_Customer'])
data['Education'],data['Marital_Status'] = data['Education'].astype('category'), data['Marital_Status'].astype('category')
data['year_month'] = data['Dt_Customer'].dt.strftime('%Y-%m')
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2240 entries, 0 to 2239
    Data columns (total 31 columns):
     #   Column               Non-Null Count  Dtype         
    ---  ------               --------------  -----         
     0   ID                   2240 non-null   int64         
     1   Year_Birth           2240 non-null   int64         
     2   Education            2240 non-null   category      
     3   Marital_Status       2240 non-null   category      
     4   Income               2216 non-null   float64       
     5   Kidhome              2240 non-null   int64         
     6   Teenhome             2240 non-null   int64         
     7   Dt_Customer          2240 non-null   datetime64[ns]
     8   Recency              2240 non-null   int64         
     9   MntWines             2240 non-null   int64         
     10  MntFruits            2240 non-null   int64         
     11  MntMeatProducts      2240 non-null   int64         
     12  MntFishProducts      2240 non-null   int64         
     13  MntSweetProducts     2240 non-null   int64         
     14  MntGoldProds         2240 non-null   int64         
     15  NumDealsPurchases    2240 non-null   int64         
     16  NumWebPurchases      2240 non-null   int64         
     17  NumCatalogPurchases  2240 non-null   int64         
     18  NumStorePurchases    2240 non-null   int64         
     19  NumWebVisitsMonth    2240 non-null   int64         
     20  AcceptedCmp3         2240 non-null   int64         
     21  AcceptedCmp4         2240 non-null   int64         
     22  AcceptedCmp5         2240 non-null   int64         
     23  AcceptedCmp1         2240 non-null   int64         
     24  AcceptedCmp2         2240 non-null   int64         
     25  Complain             2240 non-null   int64         
     26  Z_CostContact        2240 non-null   int64         
     27  Z_Revenue            2240 non-null   int64         
     28  Response             2240 non-null   int64         
     29  mnt_products         2240 non-null   int64         
     30  year_month           2240 non-null   object        
    dtypes: category(2), datetime64[ns](1), float64(1), int64(26), object(1)
    memory usage: 512.6+ KB
    


```python
#see variance to see if need to drop some values
data.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ID</th>
      <td>2240.0</td>
      <td>5592.159821</td>
      <td>3246.662198</td>
      <td>0.0</td>
      <td>2828.25</td>
      <td>5458.5</td>
      <td>8427.75</td>
      <td>11191.0</td>
    </tr>
    <tr>
      <th>Year_Birth</th>
      <td>2240.0</td>
      <td>1968.805804</td>
      <td>11.984069</td>
      <td>1893.0</td>
      <td>1959.00</td>
      <td>1970.0</td>
      <td>1977.00</td>
      <td>1996.0</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>2216.0</td>
      <td>52247.251354</td>
      <td>25173.076661</td>
      <td>1730.0</td>
      <td>35303.00</td>
      <td>51381.5</td>
      <td>68522.00</td>
      <td>666666.0</td>
    </tr>
    <tr>
      <th>Kidhome</th>
      <td>2240.0</td>
      <td>0.444196</td>
      <td>0.538398</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Teenhome</th>
      <td>2240.0</td>
      <td>0.506250</td>
      <td>0.544538</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>Recency</th>
      <td>2240.0</td>
      <td>49.109375</td>
      <td>28.962453</td>
      <td>0.0</td>
      <td>24.00</td>
      <td>49.0</td>
      <td>74.00</td>
      <td>99.0</td>
    </tr>
    <tr>
      <th>MntWines</th>
      <td>2240.0</td>
      <td>303.935714</td>
      <td>336.597393</td>
      <td>0.0</td>
      <td>23.75</td>
      <td>173.5</td>
      <td>504.25</td>
      <td>1493.0</td>
    </tr>
    <tr>
      <th>MntFruits</th>
      <td>2240.0</td>
      <td>26.302232</td>
      <td>39.773434</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>8.0</td>
      <td>33.00</td>
      <td>199.0</td>
    </tr>
    <tr>
      <th>MntMeatProducts</th>
      <td>2240.0</td>
      <td>166.950000</td>
      <td>225.715373</td>
      <td>0.0</td>
      <td>16.00</td>
      <td>67.0</td>
      <td>232.00</td>
      <td>1725.0</td>
    </tr>
    <tr>
      <th>MntFishProducts</th>
      <td>2240.0</td>
      <td>37.525446</td>
      <td>54.628979</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>12.0</td>
      <td>50.00</td>
      <td>259.0</td>
    </tr>
    <tr>
      <th>MntSweetProducts</th>
      <td>2240.0</td>
      <td>27.062946</td>
      <td>41.280498</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>8.0</td>
      <td>33.00</td>
      <td>263.0</td>
    </tr>
    <tr>
      <th>MntGoldProds</th>
      <td>2240.0</td>
      <td>44.021875</td>
      <td>52.167439</td>
      <td>0.0</td>
      <td>9.00</td>
      <td>24.0</td>
      <td>56.00</td>
      <td>362.0</td>
    </tr>
    <tr>
      <th>NumDealsPurchases</th>
      <td>2240.0</td>
      <td>2.325000</td>
      <td>1.932238</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>3.00</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>NumWebPurchases</th>
      <td>2240.0</td>
      <td>4.084821</td>
      <td>2.778714</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>4.0</td>
      <td>6.00</td>
      <td>27.0</td>
    </tr>
    <tr>
      <th>NumCatalogPurchases</th>
      <td>2240.0</td>
      <td>2.662054</td>
      <td>2.923101</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
      <td>4.00</td>
      <td>28.0</td>
    </tr>
    <tr>
      <th>NumStorePurchases</th>
      <td>2240.0</td>
      <td>5.790179</td>
      <td>3.250958</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>5.0</td>
      <td>8.00</td>
      <td>13.0</td>
    </tr>
    <tr>
      <th>NumWebVisitsMonth</th>
      <td>2240.0</td>
      <td>5.316518</td>
      <td>2.426645</td>
      <td>0.0</td>
      <td>3.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>20.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp3</th>
      <td>2240.0</td>
      <td>0.072768</td>
      <td>0.259813</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp4</th>
      <td>2240.0</td>
      <td>0.074554</td>
      <td>0.262728</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp5</th>
      <td>2240.0</td>
      <td>0.072768</td>
      <td>0.259813</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp1</th>
      <td>2240.0</td>
      <td>0.064286</td>
      <td>0.245316</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>AcceptedCmp2</th>
      <td>2240.0</td>
      <td>0.013393</td>
      <td>0.114976</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Complain</th>
      <td>2240.0</td>
      <td>0.009375</td>
      <td>0.096391</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Z_CostContact</th>
      <td>2240.0</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>Z_Revenue</th>
      <td>2240.0</td>
      <td>11.000000</td>
      <td>0.000000</td>
      <td>11.0</td>
      <td>11.00</td>
      <td>11.0</td>
      <td>11.00</td>
      <td>11.0</td>
    </tr>
    <tr>
      <th>Response</th>
      <td>2240.0</td>
      <td>0.149107</td>
      <td>0.356274</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>mnt_products</th>
      <td>2240.0</td>
      <td>605.798214</td>
      <td>602.249288</td>
      <td>5.0</td>
      <td>68.75</td>
      <td>396.0</td>
      <td>1045.50</td>
      <td>2525.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Z_CostContact and Z_Revenue have no variance so drop
data.drop(columns=['Z_Revenue', 'Z_CostContact'], inplace=True)
```


```python
#check for missing values
data.isna().sum()
```




    ID                      0
    Year_Birth              0
    Education               0
    Marital_Status          0
    Income                 24
    Kidhome                 0
    Teenhome                0
    Dt_Customer             0
    Recency                 0
    MntWines                0
    MntFruits               0
    MntMeatProducts         0
    MntFishProducts         0
    MntSweetProducts        0
    MntGoldProds            0
    NumDealsPurchases       0
    NumWebPurchases         0
    NumCatalogPurchases     0
    NumStorePurchases       0
    NumWebVisitsMonth       0
    AcceptedCmp3            0
    AcceptedCmp4            0
    AcceptedCmp5            0
    AcceptedCmp1            0
    AcceptedCmp2            0
    Complain                0
    Response                0
    mnt_products            0
    year_month              0
    dtype: int64




```python
#need to check on outliers to filter out
def boxplots_custom(dataset, columns_list, rows, cols, suptitle,size=(20,16)):
    fig, axs = plt.subplots(rows, cols,  figsize=size)
    fig.suptitle(suptitle,y=0.93, size=16)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        if i % cols == 0:
            axs[i].set_ylabel('Values')
        sns.boxplot( data=dataset[data], orient='v', ax=axs[i], palette = 'cool')
        axs[i].set_title(data)
        
boxplots_custom(dataset=data, columns_list= ['Income', 'Kidhome','Teenhome', 'Recency', 'MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth'], rows=3, cols=5, suptitle='Boxplots For Outlier Detection')
```


    
![png](output_11_0.png)
    



```python
NumericalOutlierColumns = ['Income', 'MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
Q1 = data[NumericalOutlierColumns].quantile(0.0)
Q3 = data[NumericalOutlierColumns].quantile(0.90)
IQR = Q3 - Q1
print('Interquartile Range for Variability of Median\n',IQR)

filtered_data = data[~((data[NumericalOutlierColumns] < (Q1 - 1.5 * IQR)) |(data[NumericalOutlierColumns] > (Q3 + 1.5 * IQR))).any(axis=1)]
display(data.shape)
display(filtered_data.shape)
```

    Interquartile Range for Variability of Median
     Income                 78114.0
    MntWines                 822.1
    MntFruits                 83.0
    MntMeatProducts          499.0
    MntFishProducts          120.0
    MntSweetProducts          89.0
    MntGoldProds             122.0
    NumDealsPurchases          5.0
    NumWebPurchases            8.0
    NumCatalogPurchases        7.0
    NumStorePurchases         11.0
    NumWebVisitsMonth          8.0
    dtype: float64
    


    (2240, 29)



    (2223, 29)



```python
#See if outliers were taken out 
NumericalOutlierColumns = ['Income', 'MntWines', 'MntFruits','MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
            'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth']
def boxplots_custom(dataset, columns_list, rows, cols, suptitle,size=(22,20)):
    fig, axs = plt.subplots(rows, cols,  figsize=size)
    fig.suptitle(suptitle,y=0.93, size=16)
    axs = axs.flatten()
    for i, data in enumerate(columns_list):
        if i % cols == 0:
            axs[i].set_ylabel('Values')
        sns.boxplot( data=dataset[data], orient='v', ax=axs[i], palette = 'cool')
        axs[i].set_title(data)
        
boxplots_custom(dataset=filtered_data, columns_list= NumericalOutlierColumns, rows=3, cols=4, suptitle='Boxplots After Outlier Removal')
```


    
![png](output_13_0.png)
    



```python
data2 = filtered_data
print('Skewness before filling nans', round(data2['Income'].skew(),3))

data2["Income"] = data2["Income"].fillna(data.groupby(['Education', 'Kidhome'])["Income"].transform('mean')) #filling nans

data2['Income'].plot(kind='hist', bins=30)
plt.title('Skewness after filling nans')
plt.grid();
print('Skewness after filling nans', round(data2['Income'].skew(),3))
```

    Skewness before filling nans 0.228
    Skewness after filling nans 0.228
    


    
![png](output_14_1.png)
    



```python
#confirm NaN is gone after filter of outliers
data.isna().sum()
```




    ID                      0
    Year_Birth              0
    Education               0
    Marital_Status          0
    Income                 24
    Kidhome                 0
    Teenhome                0
    Dt_Customer             0
    Recency                 0
    MntWines                0
    MntFruits               0
    MntMeatProducts         0
    MntFishProducts         0
    MntSweetProducts        0
    MntGoldProds            0
    NumDealsPurchases       0
    NumWebPurchases         0
    NumCatalogPurchases     0
    NumStorePurchases       0
    NumWebVisitsMonth       0
    AcceptedCmp3            0
    AcceptedCmp4            0
    AcceptedCmp5            0
    AcceptedCmp1            0
    AcceptedCmp2            0
    Complain                0
    Response                0
    mnt_products            0
    year_month              0
    dtype: int64




```python
#Organizing Data
        #will want to update education to american standards ✓
        #will want to create an age as it currently only shows birth year & categorical dimension✓
        #will want to have relationship be boolean to "Do they have a partner?"✓
        #will want to count teens as children 
        #will want to rename products, campaigns, and ways products were purchased as the naming convention currently is not clear

```


```python
print('The last day a consumer was enrolled is ', data['Dt_Customer'].dt.date.max())
```

    The last day a consumer was enrolled is  2014-12-06
    


```python
#Make Age Categorical
data3 = data2
data3.rename(columns = {'Year_Birth':'Age'}, inplace = True)
data3['Age'] = data3.Age.apply(lambda x: 2021-x)
bins = [0, 35, 65, np.inf]
names = ['Young Age', 'Middle Age', 'Senior']
data3['Categorical_Age'] = pd.cut(data2['Age'], bins, labels=names)
data3.insert(2, 'CustomerFor', (np.datetime64('2016-12-07') - data['Dt_Customer']).dt.days)

#New Metric For Children as Boolean
data3['NumChildren'] = data2['Kidhome'] + data2['Teenhome']
data3['HasChildren'] = data2["NumChildren"].replace({0: 'No', 
                                                      1: 'Yes',
                                                      2: 'Yes',
                                                      3: 'Yes'})
#New Metric For Relationship as Boolean
data3['Marital_Status'].replace(['YOLO', 'Absurd', 'Alone'], 'Single', inplace=True)
data3['HasAPartner'] = data2["Marital_Status"].replace({'Single': 'No', 
                                                      'Widow': 'No',
                                                      'Divorced': 'No',
                                                      'Together': 'Yes',
                                                      'Married': 'Yes'})
#Education Update
data3['Education'].replace(['Basic','2n Cycle', 'Graduation'], 
                          ['Below Bachelor', 'Master', 'Bachelor'], inplace=True)

data3['MntTotal'] = data3.filter(like='Mnt').sum(axis=1)
#adding more metrics
data3['AvgWeb'] = round(data3['NumWebPurchases']/data3['NumWebVisitsMonth'],2)
data3.fillna({'AvgWeb' : 0},inplace=True) # Handling for cases where division by 0 may yield unwanted results
data3.replace(np.inf,0,inplace=True)

  
data3['NumTotal'] = np.sum(data3.filter(regex='Purchases'), axis=1)
for i in data3.filter(regex='Purchases').columns:
    if(i!='NumTotal'):
        data3[i] = round((data2[i]*100)/data3['NumTotal'],2)
        data3.fillna({i : 0},inplace=True)
data3.drop(columns=['NumTotal'],inplace=True)

data3['Expenses'] = data3['MntWines'] + data3['MntFruits'] + data3['MntMeatProducts'] + data3['MntFishProducts'] + data3['MntSweetProducts'] + data3['MntGoldProds']
data3['TotalAcceptedCmp'] = data3['AcceptedCmp1'] + data3['AcceptedCmp2'] + data3['AcceptedCmp3'] + data3['AcceptedCmp4'] + data3['AcceptedCmp5'] + data3['Response']
data3['NumTotalPurchases'] = data3['NumWebPurchases'] + data3['NumCatalogPurchases'] + data3['NumStorePurchases'] + data3['NumDealsPurchases']
data3['ExpensePer'] = round((data3['MntTotal']*100) / data3['Income'],2)

from datetime import date
today = date.today()

data3.rename(columns = {'Dt_Customer':'TotalEnrollDays'}, inplace = True)
data3['TotalEnrollDays'] = pd.to_datetime(today) - data3['TotalEnrollDays']
data3['TotalEnrollDays'] = [float(str(data3['TotalEnrollDays'][x])[:4]) for x in data2.index]
data3['TotalEnrollDays'] = round(data3['TotalEnrollDays']/365,2)
data3.rename(columns = {'TotalEnrollDays':'TotalEnrollYrs'}, inplace = True)

data3['NumAllPurchases'] = data3['NumWebPurchases']+data3['NumCatalogPurchases']+data3['NumStorePurchases']
data3['AverageCheck'] = round((data3['MntTotal'] / data3['NumAllPurchases']), 1)

data3
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>CustomerFor</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>TotalEnrollYrs</th>
      <th>Recency</th>
      <th>...</th>
      <th>HasChildren</th>
      <th>HasAPartner</th>
      <th>MntTotal</th>
      <th>AvgWeb</th>
      <th>Expenses</th>
      <th>TotalAcceptedCmp</th>
      <th>NumTotalPurchases</th>
      <th>ExpensePer</th>
      <th>NumAllPurchases</th>
      <th>AverageCheck</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>64</td>
      <td>1703</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.63</td>
      <td>58</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>1617</td>
      <td>1.14</td>
      <td>1617</td>
      <td>1</td>
      <td>100.00</td>
      <td>2.78</td>
      <td>88.00</td>
      <td>18.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>67</td>
      <td>857</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.31</td>
      <td>38</td>
      <td>...</td>
      <td>Yes</td>
      <td>No</td>
      <td>27</td>
      <td>0.20</td>
      <td>27</td>
      <td>0</td>
      <td>100.00</td>
      <td>0.06</td>
      <td>66.67</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>56</td>
      <td>1204</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.26</td>
      <td>26</td>
      <td>...</td>
      <td>No</td>
      <td>Yes</td>
      <td>776</td>
      <td>2.00</td>
      <td>776</td>
      <td>0</td>
      <td>100.00</td>
      <td>1.08</td>
      <td>95.24</td>
      <td>8.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>37</td>
      <td>797</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.15</td>
      <td>26</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>53</td>
      <td>0.33</td>
      <td>53</td>
      <td>0</td>
      <td>100.00</td>
      <td>0.20</td>
      <td>75.00</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>40</td>
      <td>1053</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.85</td>
      <td>94</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>422</td>
      <td>1.00</td>
      <td>422</td>
      <td>0</td>
      <td>100.01</td>
      <td>0.72</td>
      <td>73.69</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>10870</td>
      <td>54</td>
      <td>1273</td>
      <td>Bachelor</td>
      <td>Married</td>
      <td>61223.0</td>
      <td>0</td>
      <td>1</td>
      <td>8.45</td>
      <td>46</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>1341</td>
      <td>1.80</td>
      <td>1341</td>
      <td>0</td>
      <td>100.00</td>
      <td>2.19</td>
      <td>88.89</td>
      <td>15.1</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>4001</td>
      <td>75</td>
      <td>793</td>
      <td>PhD</td>
      <td>Together</td>
      <td>64014.0</td>
      <td>2</td>
      <td>1</td>
      <td>7.13</td>
      <td>56</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>444</td>
      <td>1.14</td>
      <td>444</td>
      <td>1</td>
      <td>100.00</td>
      <td>0.69</td>
      <td>68.18</td>
      <td>6.5</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>7270</td>
      <td>40</td>
      <td>1047</td>
      <td>Bachelor</td>
      <td>Divorced</td>
      <td>56981.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.83</td>
      <td>91</td>
      <td>...</td>
      <td>No</td>
      <td>No</td>
      <td>1241</td>
      <td>0.33</td>
      <td>1241</td>
      <td>1</td>
      <td>100.00</td>
      <td>2.18</td>
      <td>94.74</td>
      <td>13.1</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>8235</td>
      <td>65</td>
      <td>1048</td>
      <td>Master</td>
      <td>Together</td>
      <td>69245.0</td>
      <td>0</td>
      <td>1</td>
      <td>7.83</td>
      <td>8</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>843</td>
      <td>2.00</td>
      <td>843</td>
      <td>0</td>
      <td>100.01</td>
      <td>1.22</td>
      <td>91.31</td>
      <td>9.2</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>9405</td>
      <td>67</td>
      <td>1514</td>
      <td>PhD</td>
      <td>Married</td>
      <td>52869.0</td>
      <td>1</td>
      <td>1</td>
      <td>9.11</td>
      <td>40</td>
      <td>...</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>172</td>
      <td>0.43</td>
      <td>172</td>
      <td>1</td>
      <td>99.99</td>
      <td>0.33</td>
      <td>72.72</td>
      <td>2.4</td>
    </tr>
  </tbody>
</table>
<p>2223 rows × 42 columns</p>
</div>




```python
#Education Percentage

Cool = sns.color_palette('cool')[0:5]

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x= data3['Education'], palette=Cool, data=data2)
plt.xlabel("Education", size=16, color= 'white')
plt.ylabel("Count", size=16, color= 'white')
plt.title("Consumer Education", size=24, color= 'white')

fig, ax = plt.subplots(figsize=(10, 8))
porportion = dict(data3['Education'].value_counts())
patches, texts, pcts = ax.pie(porportion.values(), labels=porportion.keys(), autopct ='% .1f%%', pctdistance=0.75,
    wedgeprops={'linewidth': 2.5, 'edgecolor': '#778899'},
    textprops={'size': 'x-large'},
    startangle=90,
    colors = Cool)

for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='#3A3A3A', fontweight='bold', fontsize = 22)
plt.setp(texts, fontweight=600, fontsize = 25)
plt.legend(patches, labels=porportion.keys(), loc="upper left")
ax.set_title('Consumer Education Breakdown', fontsize=30, color= 'white')

plt.tight_layout()
plt.show()
```


    
![png](output_19_0.png)
    



    
![png](output_19_1.png)
    



```python
#Relationship Percentage

porportion = dict(data3['Marital_Status'].value_counts())
Cool = sns.color_palette('cool')[0:5]

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x= data3['Marital_Status'], palette=Cool, data=data2)
plt.xlabel("Marital Status", size=16, color= 'white')
plt.ylabel("Count", size=16, color= 'white')
plt.title("Consumer Relationship", size=24, color= 'white')

fig, ax = plt.subplots(figsize=(10,8))
patches, texts, pcts = ax.pie(porportion.values(), labels=porportion.keys(), autopct ='% .1f%%', pctdistance=0.75,
    wedgeprops={'linewidth': 2.5, 'edgecolor': '#778899'},
    textprops={'size': 'x-large'},
    startangle=90,
    colors = Cool)
# For each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='#3A3A3A', fontweight='bold', fontsize = 22)
plt.setp(texts, fontweight=600, fontsize = 25)
plt.legend(patches, labels=porportion.keys(), loc="upper left")
ax.set_title('Consumer Relationship Breakdown', fontsize=30, color= 'white')

plt.tight_layout()
plt.show()
```


    
![png](output_20_0.png)
    



    
![png](output_20_1.png)
    



```python
#Age Percentage
porportion = dict(data3['Categorical_Age'].value_counts())
Cool = sns.color_palette('cool')[0:5]

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x= data3['Categorical_Age'], palette=Cool, data=data2)
plt.xlabel("Categorical Age", size=16, color= 'white')
plt.ylabel("Count", size=16, color= 'white')
plt.title("Consumer Age", size=24, color= 'white')

fig, ax = plt.subplots(figsize=(10,8))
patches, texts, pcts = ax.pie(porportion.values(), labels=porportion.keys(), autopct ='% .1f%%', pctdistance=0.75,
    wedgeprops={'linewidth': 2.5, 'edgecolor': '#778899'},
    textprops={'size': 'x-large'},
    startangle=90,
    colors = Cool)
# For each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='#3A3A3A', fontweight='bold', fontsize = 22)
plt.setp(texts, fontweight=600, fontsize = 25)
plt.legend(patches, labels=porportion.keys(), loc="upper left")
ax.set_title('Consumer Age Breakdown', fontsize=30, color= 'white')

plt.tight_layout()
plt.show()
```


    
![png](output_21_0.png)
    



    
![png](output_21_1.png)
    



```python
#Child Percentage
porportion = dict(data3['NumChildren'].value_counts())
Cool = sns.color_palette('cool')[0:5]

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x= data3['NumChildren'], palette=Cool, data=data2)
plt.xlabel("Number of Children", size=16, color= 'white')
plt.ylabel("Consumer Count", size=16, color= 'white')
plt.title("Consumers Children Count", size=24, color= 'white')

fig, ax = plt.subplots(figsize=(10,8))
patches, texts, pcts = ax.pie(porportion.values(), labels=porportion.keys(), autopct ='% .1f%%', pctdistance=0.75,
    wedgeprops={'linewidth': 2.5, 'edgecolor': '#778899'},
    textprops={'size': 'x-large'},
    startangle=90,
    colors = Cool)
# For each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='#3A3A3A', fontweight='bold', fontsize = 22)
plt.setp(texts, fontweight=600, fontsize = 25)
plt.legend(patches, labels=porportion.keys(), loc="upper left")
ax.set_title('Consumer Child Amount Breakdown', fontsize=30, color= 'white')

plt.tight_layout()
plt.show()
```


    
![png](output_22_0.png)
    



    
![png](output_22_1.png)
    



```python
#Product Percentage
mnt = data.filter(like='Mnt').apply(lambda x: sum(x), axis=0)
porportion = dict(mnt)
Cool = sns.color_palette('cool')[0:5]
fig, ax = plt.subplots(figsize=(12,10))
patches, texts, pcts = ax.pie(porportion.values(), labels=['Wine', 'Fruits', 'Meat','Fish', 'Sweets', 'Gold'], 
                        autopct ='% .1f%%', 
                        pctdistance=0.75,
                        wedgeprops={'linewidth': 2.5, 'edgecolor': '#778899'},
                        textprops={'size': 'x-large'},
                        startangle=90,
                        colors = Cool)
# For each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='#3A3A3A', fontweight='bold', fontsize = 22)
plt.setp(texts, fontweight=600, fontsize = 25)
plt.legend(patches, labels=['Wine', 'Fruits', 'Meat','Fish', 'Sweets', 'Gold'], loc="upper left")
ax.set_title('Consumer Consumption Breakdown', fontsize=30, color= 'white')
plt.tight_layout()
plt.show()
```


    
![png](output_23_0.png)
    



```python
#Source Percentage
num = data.filter(regex='Num[^Deals].+Purchases').sum(axis=0)
porportion = dict(num)
Cool = sns.color_palette('cool')[0:5]
fig, ax = plt.subplots(figsize=(12,10))

patches, texts, pcts = ax.pie(porportion.values(), labels=['Website', 'Catalog', 'Store'], 
                        autopct ='% .1f%%', 
                        pctdistance=0.75,
                        wedgeprops={'linewidth': 2.5, 'edgecolor': '#778899'},
                        textprops={'size': 'x-large'},
                        startangle=90,
                        colors = Cool)
# For each wedge, set the corresponding text label color to the wedge's
# face color.
for i, patch in enumerate(patches):
  texts[i].set_color(patch.get_facecolor())
plt.setp(pcts, color='#3A3A3A', fontweight='bold', fontsize = 22)
plt.setp(texts, fontweight=600, fontsize = 25)
plt.legend(patches, labels=['Website', 'Catalog', 'Store'], loc="upper left")
ax.set_title('Consumer Source Breakdown', fontsize=30, color= 'white')
plt.tight_layout()
plt.show()
```


    
![png](output_24_0.png)
    



```python
#Product Purchase Total
data3 = data3.assign(
        percentWines=lambda x: x['MntWines'] / x['MntTotal'] * 100,
        percentMeat=lambda x: x['MntMeatProducts'] / x['MntTotal'] * 100,
        percentFruits=lambda x: x['MntFruits'] / x['MntTotal'] * 100,
        percentFish=lambda x: x['MntFishProducts'] / x['MntTotal'] * 100,
        percentSweets=lambda x: x['MntSweetProducts'] / x['MntTotal'] * 100,
        percentGold=lambda x: x['MntGoldProds'] / x['MntTotal'] * 100,
)
data3.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Age</th>
      <th>CustomerFor</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>TotalEnrollYrs</th>
      <th>Recency</th>
      <th>...</th>
      <th>NumTotalPurchases</th>
      <th>ExpensePer</th>
      <th>NumAllPurchases</th>
      <th>AverageCheck</th>
      <th>percentWines</th>
      <th>percentMeat</th>
      <th>percentFruits</th>
      <th>percentFish</th>
      <th>percentSweets</th>
      <th>percentGold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5524</td>
      <td>64</td>
      <td>1703</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.63</td>
      <td>58</td>
      <td>...</td>
      <td>100.00</td>
      <td>2.78</td>
      <td>88.00</td>
      <td>18.4</td>
      <td>39.270254</td>
      <td>33.766234</td>
      <td>5.442177</td>
      <td>10.636982</td>
      <td>5.442177</td>
      <td>5.442177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2174</td>
      <td>67</td>
      <td>857</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.31</td>
      <td>38</td>
      <td>...</td>
      <td>100.00</td>
      <td>0.06</td>
      <td>66.67</td>
      <td>0.4</td>
      <td>40.740741</td>
      <td>22.222222</td>
      <td>3.703704</td>
      <td>7.407407</td>
      <td>3.703704</td>
      <td>22.222222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4141</td>
      <td>56</td>
      <td>1204</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.26</td>
      <td>26</td>
      <td>...</td>
      <td>100.00</td>
      <td>1.08</td>
      <td>95.24</td>
      <td>8.1</td>
      <td>54.896907</td>
      <td>16.365979</td>
      <td>6.314433</td>
      <td>14.304124</td>
      <td>2.706186</td>
      <td>5.412371</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6182</td>
      <td>37</td>
      <td>797</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.15</td>
      <td>26</td>
      <td>...</td>
      <td>100.00</td>
      <td>0.20</td>
      <td>75.00</td>
      <td>0.7</td>
      <td>20.754717</td>
      <td>37.735849</td>
      <td>7.547170</td>
      <td>18.867925</td>
      <td>5.660377</td>
      <td>9.433962</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5324</td>
      <td>40</td>
      <td>1053</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.85</td>
      <td>94</td>
      <td>...</td>
      <td>100.01</td>
      <td>0.72</td>
      <td>73.69</td>
      <td>5.7</td>
      <td>40.995261</td>
      <td>27.962085</td>
      <td>10.189573</td>
      <td>10.900474</td>
      <td>6.398104</td>
      <td>3.554502</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 48 columns</p>
</div>




```python
#Product Purchase & Education
fig, axes = plt.subplots(4, 6, figsize=(16, 15), sharey=True)
fig.suptitle('Product Purchase Percentage by Education', fontsize=20)
Cool = sns.color_palette('cool')[0:5]

for i, value in enumerate(data3['Education'].unique()):
    sns.boxplot(data=data3.query(f'Education == "{value}"'), y='percentWines', showfliers=False, color=Cool[i], ax=axes[i, 0])
    axes[i, 0].set_ylim(0, 100)
    axes[i, 0].set_xlabel('Wine', color = 'White')
    axes[i, 0].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Education == "{value}"'), y='percentMeat', showfliers=False, color=Cool[i], ax=axes[i, 1])
    axes[i, 1].set_xlabel('Meat', color = 'White')
    axes[i, 1].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Education == "{value}"'), y='percentFruits', showfliers=False, color=Cool[i], ax=axes[i, 2])
    axes[i, 2].set_xlabel('Fruits', color = 'White')
    axes[i, 2].set_ylabel('', color = 'White')
    axes[i, 2].set_title(f'{value}', x=1, color = 'White')
    
    sns.boxplot(data=data3.query(f'Education == "{value}"'), y='percentFish', showfliers=False, color=Cool[i], ax=axes[i, 3])
    axes[i, 3].set_xlabel('Fish', color = 'White')
    axes[i, 3].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Education == "{value}"'), y='percentSweets', showfliers=False, color=Cool[i], ax=axes[i, 4])
    axes[i, 4].set_xlabel('Sweets', color = 'White')
    axes[i, 4].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Education == "{value}"'), y='percentGold', showfliers=False, color=Cool[i], ax=axes[i, 5])
    axes[i, 5].set_xlabel('Gold', color = 'White')
    axes[i, 5].set_ylabel('', color = 'White')
    
plt.tight_layout()
```


    
![png](output_26_0.png)
    



```python
#Product Purchase & Relationship
fig, axes = plt.subplots(5, 6, figsize=(16, 15), sharey=True)
fig.suptitle('Product Purchase Percentage by Marital Status', fontsize=20)
Cool = sns.color_palette('cool')[0:5]

for i, value in enumerate(data3['Marital_Status'].unique()):
    sns.boxplot(data=data3.query(f'Marital_Status == "{value}"'), y='percentWines', showfliers=False, color=Cool[i], ax=axes[i, 0])
    axes[i, 0].set_ylim(0, 100)
    axes[i, 0].set_xlabel('Wine', color = 'White')
    axes[i, 0].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Marital_Status== "{value}"'), y='percentMeat', showfliers=False, color=Cool[i], ax=axes[i, 1])
    axes[i, 1].set_xlabel('Meat', color = 'White')
    axes[i, 1].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Marital_Status == "{value}"'), y='percentFruits', showfliers=False, color=Cool[i], ax=axes[i, 2])
    axes[i, 2].set_xlabel('Fruits', color = 'White')
    axes[i, 2].set_ylabel('', color = 'White')
    axes[i, 2].set_title(f'{value}', x=1, color = 'White')
    
    sns.boxplot(data=data3.query(f'Marital_Status == "{value}"'), y='percentFish', showfliers=False, color=Cool[i], ax=axes[i, 3])
    axes[i, 3].set_xlabel('Fish', color = 'White')
    axes[i, 3].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Marital_Status == "{value}"'), y='percentSweets', showfliers=False, color=Cool[i], ax=axes[i, 4])
    axes[i, 4].set_xlabel('Sweets', color = 'White')
    axes[i, 4].set_ylabel('', color = 'White')
    
    sns.boxplot(data=data3.query(f'Marital_Status == "{value}"'), y='percentGold', showfliers=False, color=Cool[i], ax=axes[i, 5])
    axes[i, 5].set_xlabel('Gold', color = 'White')
    axes[i, 5].set_ylabel('', color = 'White')
    
plt.tight_layout()
```


    
![png](output_27_0.png)
    



```python
#Relationship between varaibles
g = sns.catplot(x="Education", y="Income",
                hue="HasChildren", col="HasAPartner",
                data=data3, kind="violin", split=True, palette = 'cool')
g.set_axis_labels("Education", "Income",size=14, color = 'White')
g.set_xticklabels(label=[ "Below Bachelor", "Bachelor", "Master", "PhD"])
g.set_titles(col_template="{col_name} Partner", row_template="{row_name}", size=16, color = 'White')

g = sns.catplot(x="Education", y="NumWebPurchases",
                hue="HasChildren", col="HasAPartner",
                data=data3, kind="violin", split=True, palette = 'cool')
g.set_axis_labels("Education", "Web Purchases",size=14, color = 'White')
g.set_xticklabels(label=[ "Below Bachelor", "Bachelor", "Master", "PhD"])
g.set_titles(col_template="{col_name} Partner", row_template="{row_name}", size=16, color = 'White')

g = sns.catplot(x="Education", y="NumStorePurchases",
                hue="HasChildren", col="HasAPartner",
                data=data3, kind="violin", split=True, palette = 'cool')
g.set_axis_labels("Education", "Store Purchases",size=14, color = 'White')
g.set_xticklabels(label=[ "Below Bachelor", "Bachelor", "Master", "PhD"])
g.set_titles(col_template="{col_name} Partner", row_template="{row_name}", size=16, color = 'White')

g = sns.catplot(x="Education", y="NumCatalogPurchases",
                hue="HasChildren", col="HasAPartner",
                data=data3, kind="violin", split=True, palette = 'cool')
g.set_axis_labels("Education", "Catalog Purchases",size=14, color = 'White')
g.set_xticklabels(label=[ "Below Bachelor", "Bachelor", "Master", "PhD"])
g.set_titles(col_template="{col_name} Partner", row_template="{row_name}", size=16, color = 'White')

g = sns.catplot(x="Education", y="TotalAcceptedCmp",
                hue="HasAPartner", col="HasChildren",
                data=data3, kind="violin", split=True, palette = 'cool')
g.set_axis_labels("Education", "Campaign Acceptance",size=14, color = 'White')
g.set_xticklabels(label=[ "Below Bachelor", "Bachelor", "Master", "PhD"])
g.set_titles(col_template="{col_name} Children", row_template="{row_name}", size=16, color = 'White')
```




    <seaborn.axisgrid.FacetGrid at 0x1a6be524af0>




    
![png](output_28_1.png)
    



    
![png](output_28_2.png)
    



    
![png](output_28_3.png)
    



    
![png](output_28_4.png)
    



    
![png](output_28_5.png)
    



```python
#ANALYSIS
data4 = data3
data4.drop(['ID'], axis = 1, inplace = True)
data4
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>CustomerFor</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>TotalEnrollYrs</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>NumTotalPurchases</th>
      <th>ExpensePer</th>
      <th>NumAllPurchases</th>
      <th>AverageCheck</th>
      <th>percentWines</th>
      <th>percentMeat</th>
      <th>percentFruits</th>
      <th>percentFish</th>
      <th>percentSweets</th>
      <th>percentGold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64</td>
      <td>1703</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.63</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>100.00</td>
      <td>2.78</td>
      <td>88.00</td>
      <td>18.4</td>
      <td>39.270254</td>
      <td>33.766234</td>
      <td>5.442177</td>
      <td>10.636982</td>
      <td>5.442177</td>
      <td>5.442177</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>857</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.31</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>100.00</td>
      <td>0.06</td>
      <td>66.67</td>
      <td>0.4</td>
      <td>40.740741</td>
      <td>22.222222</td>
      <td>3.703704</td>
      <td>7.407407</td>
      <td>3.703704</td>
      <td>22.222222</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>1204</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.26</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>100.00</td>
      <td>1.08</td>
      <td>95.24</td>
      <td>8.1</td>
      <td>54.896907</td>
      <td>16.365979</td>
      <td>6.314433</td>
      <td>14.304124</td>
      <td>2.706186</td>
      <td>5.412371</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>797</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.15</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>100.00</td>
      <td>0.20</td>
      <td>75.00</td>
      <td>0.7</td>
      <td>20.754717</td>
      <td>37.735849</td>
      <td>7.547170</td>
      <td>18.867925</td>
      <td>5.660377</td>
      <td>9.433962</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1053</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.85</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>100.01</td>
      <td>0.72</td>
      <td>73.69</td>
      <td>5.7</td>
      <td>40.995261</td>
      <td>27.962085</td>
      <td>10.189573</td>
      <td>10.900474</td>
      <td>6.398104</td>
      <td>3.554502</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>54</td>
      <td>1273</td>
      <td>Bachelor</td>
      <td>Married</td>
      <td>61223.0</td>
      <td>0</td>
      <td>1</td>
      <td>8.45</td>
      <td>46</td>
      <td>709</td>
      <td>...</td>
      <td>100.00</td>
      <td>2.19</td>
      <td>88.89</td>
      <td>15.1</td>
      <td>52.870992</td>
      <td>13.571961</td>
      <td>3.206562</td>
      <td>3.131991</td>
      <td>8.799403</td>
      <td>18.419090</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>75</td>
      <td>793</td>
      <td>PhD</td>
      <td>Together</td>
      <td>64014.0</td>
      <td>2</td>
      <td>1</td>
      <td>7.13</td>
      <td>56</td>
      <td>406</td>
      <td>...</td>
      <td>100.00</td>
      <td>0.69</td>
      <td>68.18</td>
      <td>6.5</td>
      <td>91.441441</td>
      <td>6.756757</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.801802</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>40</td>
      <td>1047</td>
      <td>Bachelor</td>
      <td>Divorced</td>
      <td>56981.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.83</td>
      <td>91</td>
      <td>908</td>
      <td>...</td>
      <td>100.00</td>
      <td>2.18</td>
      <td>94.74</td>
      <td>13.1</td>
      <td>73.166801</td>
      <td>17.485898</td>
      <td>3.867849</td>
      <td>2.578566</td>
      <td>0.966962</td>
      <td>1.933924</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>65</td>
      <td>1048</td>
      <td>Master</td>
      <td>Together</td>
      <td>69245.0</td>
      <td>0</td>
      <td>1</td>
      <td>7.83</td>
      <td>8</td>
      <td>428</td>
      <td>...</td>
      <td>100.01</td>
      <td>1.22</td>
      <td>91.31</td>
      <td>9.2</td>
      <td>50.771056</td>
      <td>25.385528</td>
      <td>3.558719</td>
      <td>9.489917</td>
      <td>3.558719</td>
      <td>7.236062</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>67</td>
      <td>1514</td>
      <td>PhD</td>
      <td>Married</td>
      <td>52869.0</td>
      <td>1</td>
      <td>1</td>
      <td>9.11</td>
      <td>40</td>
      <td>84</td>
      <td>...</td>
      <td>99.99</td>
      <td>0.33</td>
      <td>72.72</td>
      <td>2.4</td>
      <td>48.837209</td>
      <td>35.465116</td>
      <td>1.744186</td>
      <td>1.162791</td>
      <td>0.581395</td>
      <td>12.209302</td>
    </tr>
  </tbody>
</table>
<p>2223 rows × 47 columns</p>
</div>




```python
#see variance in heatmap form without no variance columns
NUMERICAL_FEATURES = ['Age', 'Income', 'NumChildren', 'CustomerFor', 
                      'Recency', 'MntWines', 'MntTotal', 
                      'NumTotalPurchases', 'TotalAcceptedCmp', 'AverageCheck']
corr_matr = data4[NUMERICAL_FEATURES].corr(method='pearson')
plt.figure(figsize=(12,12))
sns.heatmap(corr_matr, annot=True, cmap='cool', square=True)
plt.title("Correlation Heatmap", size=16, color = 'white')
plt.show()
```


    
![png](output_30_0.png)
    



```python
filtered_data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 2223 entries, 0 to 2239
    Data columns (total 42 columns):
     #   Column               Non-Null Count  Dtype   
    ---  ------               --------------  -----   
     0   ID                   2223 non-null   int64   
     1   Age                  2223 non-null   int64   
     2   CustomerFor          2223 non-null   int64   
     3   Education            2223 non-null   object  
     4   Marital_Status       2223 non-null   category
     5   Income               2223 non-null   float64 
     6   Kidhome              2223 non-null   int64   
     7   Teenhome             2223 non-null   int64   
     8   TotalEnrollYrs       2223 non-null   float64 
     9   Recency              2223 non-null   int64   
     10  MntWines             2223 non-null   int64   
     11  MntFruits            2223 non-null   int64   
     12  MntMeatProducts      2223 non-null   int64   
     13  MntFishProducts      2223 non-null   int64   
     14  MntSweetProducts     2223 non-null   int64   
     15  MntGoldProds         2223 non-null   int64   
     16  NumDealsPurchases    2223 non-null   float64 
     17  NumWebPurchases      2223 non-null   float64 
     18  NumCatalogPurchases  2223 non-null   float64 
     19  NumStorePurchases    2223 non-null   float64 
     20  NumWebVisitsMonth    2223 non-null   int64   
     21  AcceptedCmp3         2223 non-null   int64   
     22  AcceptedCmp4         2223 non-null   int64   
     23  AcceptedCmp5         2223 non-null   int64   
     24  AcceptedCmp1         2223 non-null   int64   
     25  AcceptedCmp2         2223 non-null   int64   
     26  Complain             2223 non-null   int64   
     27  Response             2223 non-null   int64   
     28  mnt_products         2223 non-null   int64   
     29  year_month           2223 non-null   object  
     30  Categorical_Age      2223 non-null   category
     31  NumChildren          2223 non-null   int64   
     32  HasChildren          2223 non-null   object  
     33  HasAPartner          2223 non-null   object  
     34  MntTotal             2223 non-null   int64   
     35  AvgWeb               2223 non-null   float64 
     36  Expenses             2223 non-null   int64   
     37  TotalAcceptedCmp     2223 non-null   int64   
     38  NumTotalPurchases    2223 non-null   float64 
     39  ExpensePer           2223 non-null   float64 
     40  NumAllPurchases      2223 non-null   float64 
     41  AverageCheck         2223 non-null   float64 
    dtypes: category(2), float64(11), int64(25), object(4)
    memory usage: 781.3+ KB
    


```python
data5 = data4
data5['AverageCheck'] = data5['AverageCheck'].replace(np.inf, np.nan)
data5 = data5.fillna(data5.mean())
data5.isnull()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>CustomerFor</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>TotalEnrollYrs</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>NumTotalPurchases</th>
      <th>ExpensePer</th>
      <th>NumAllPurchases</th>
      <th>AverageCheck</th>
      <th>percentWines</th>
      <th>percentMeat</th>
      <th>percentFruits</th>
      <th>percentFish</th>
      <th>percentSweets</th>
      <th>percentGold</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>...</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>2223 rows × 47 columns</p>
</div>




```python
SELECTED_FEATURES = ['AverageCheck', 'Income', 'NumTotalPurchases', 'MntTotal']
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)

df = data5[SELECTED_FEATURES]
# kmeans.fit(data4[SELECTED_FEATURES])
kmeans.fit(df)

pred = kmeans.predict(df)

df['cluster'] = kmeans.labels_

df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AverageCheck</th>
      <th>Income</th>
      <th>NumTotalPurchases</th>
      <th>MntTotal</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.4</td>
      <td>58138.0</td>
      <td>100.00</td>
      <td>1617</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4</td>
      <td>46344.0</td>
      <td>100.00</td>
      <td>27</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>71613.0</td>
      <td>100.00</td>
      <td>776</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.7</td>
      <td>26646.0</td>
      <td>100.00</td>
      <td>53</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.7</td>
      <td>58293.0</td>
      <td>100.01</td>
      <td>422</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>15.1</td>
      <td>61223.0</td>
      <td>100.00</td>
      <td>1341</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>6.5</td>
      <td>64014.0</td>
      <td>100.00</td>
      <td>444</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>13.1</td>
      <td>56981.0</td>
      <td>100.00</td>
      <td>1241</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>9.2</td>
      <td>69245.0</td>
      <td>100.01</td>
      <td>843</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>2.4</td>
      <td>52869.0</td>
      <td>99.99</td>
      <td>172</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2223 rows × 5 columns</p>
</div>




```python
data6 = data5
data6['K_Cluster'] = pd.Series(pred, index=df.index)
data6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>CustomerFor</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>TotalEnrollYrs</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>AverageCheck</th>
      <th>percentWines</th>
      <th>percentMeat</th>
      <th>percentFruits</th>
      <th>percentFish</th>
      <th>percentSweets</th>
      <th>percentGold</th>
      <th>K_Cluster</th>
      <th>G_Cluster</th>
      <th>HasPartner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64</td>
      <td>1703</td>
      <td>1</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.63</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>18.4</td>
      <td>39.270254</td>
      <td>33.766234</td>
      <td>5.442177</td>
      <td>10.636982</td>
      <td>5.442177</td>
      <td>5.442177</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>857</td>
      <td>1</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.31</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>0.4</td>
      <td>40.740741</td>
      <td>22.222222</td>
      <td>3.703704</td>
      <td>7.407407</td>
      <td>3.703704</td>
      <td>22.222222</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>1204</td>
      <td>1</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.26</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>8.1</td>
      <td>54.896907</td>
      <td>16.365979</td>
      <td>6.314433</td>
      <td>14.304124</td>
      <td>2.706186</td>
      <td>5.412371</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>797</td>
      <td>1</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.15</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>0.7</td>
      <td>20.754717</td>
      <td>37.735849</td>
      <td>7.547170</td>
      <td>18.867925</td>
      <td>5.660377</td>
      <td>9.433962</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1053</td>
      <td>3</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.85</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5.7</td>
      <td>40.995261</td>
      <td>27.962085</td>
      <td>10.189573</td>
      <td>10.900474</td>
      <td>6.398104</td>
      <td>3.554502</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python
SELECTED_FEATURES = ['AverageCheck', 'Income', 'NumTotalPurchases', 'MntTotal']
gmm = GaussianMixture(n_components = 4, random_state=42)
df2 = (data5[SELECTED_FEATURES])
gmm.fit(df2)

labels = gmm.predict(df2)
frame = pd.DataFrame(df2)
frame['cluster'] = labels

df2
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>AverageCheck</th>
      <th>Income</th>
      <th>NumTotalPurchases</th>
      <th>MntTotal</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18.4</td>
      <td>58138.0</td>
      <td>100.00</td>
      <td>1617</td>
      <td>3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.4</td>
      <td>46344.0</td>
      <td>100.00</td>
      <td>27</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.1</td>
      <td>71613.0</td>
      <td>100.00</td>
      <td>776</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.7</td>
      <td>26646.0</td>
      <td>100.00</td>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.7</td>
      <td>58293.0</td>
      <td>100.01</td>
      <td>422</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>15.1</td>
      <td>61223.0</td>
      <td>100.00</td>
      <td>1341</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>6.5</td>
      <td>64014.0</td>
      <td>100.00</td>
      <td>444</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>13.1</td>
      <td>56981.0</td>
      <td>100.00</td>
      <td>1241</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>9.2</td>
      <td>69245.0</td>
      <td>100.01</td>
      <td>843</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>2.4</td>
      <td>52869.0</td>
      <td>99.99</td>
      <td>172</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>2223 rows × 5 columns</p>
</div>




```python
data6 = data5  
data6['G_Cluster'] = pd.Series(labels, index=df.index)
data6.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>CustomerFor</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>TotalEnrollYrs</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>AverageCheck</th>
      <th>percentWines</th>
      <th>percentMeat</th>
      <th>percentFruits</th>
      <th>percentFish</th>
      <th>percentSweets</th>
      <th>percentGold</th>
      <th>K_Cluster</th>
      <th>G_Cluster</th>
      <th>HasPartner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64</td>
      <td>1703</td>
      <td>1</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.63</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>18.4</td>
      <td>39.270254</td>
      <td>33.766234</td>
      <td>5.442177</td>
      <td>10.636982</td>
      <td>5.442177</td>
      <td>5.442177</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>857</td>
      <td>1</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.31</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>0.4</td>
      <td>40.740741</td>
      <td>22.222222</td>
      <td>3.703704</td>
      <td>7.407407</td>
      <td>3.703704</td>
      <td>22.222222</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>1204</td>
      <td>1</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.26</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>8.1</td>
      <td>54.896907</td>
      <td>16.365979</td>
      <td>6.314433</td>
      <td>14.304124</td>
      <td>2.706186</td>
      <td>5.412371</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>797</td>
      <td>1</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.15</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>0.7</td>
      <td>20.754717</td>
      <td>37.735849</td>
      <td>7.547170</td>
      <td>18.867925</td>
      <td>5.660377</td>
      <td>9.433962</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1053</td>
      <td>3</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.85</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5.7</td>
      <td>40.995261</td>
      <td>27.962085</td>
      <td>10.189573</td>
      <td>10.900474</td>
      <td>6.398104</td>
      <td>3.554502</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 50 columns</p>
</div>




```python

plt.figure(figsize=(8, 6))
pl = sns.countplot(x=data6["K_Cluster"], palette= 'cool')
plt.xlabel("Clusters", size=16, color= 'white')
plt.ylabel("Cluster Count", size=16, color= 'white')
plt.title("KMeans Cluster Count", size=24, color= 'white')
    
plt.figure(figsize=(8,6))
pl = sns.countplot(x=data6["G_Cluster"], palette= 'cool')
plt.xlabel("Clusters", size=16, color= 'white')
plt.ylabel("Cluster Count", size=16, color= 'white')
plt.title("GMM Cluster Count", size=24, color= 'white')

plt.show()
```


    
![png](output_37_0.png)
    



    
![png](output_37_1.png)
    



```python

data6['G_Cluster'].value_counts()
```




    3    1393
    0     824
    1       4
    2       2
    Name: G_Cluster, dtype: int64




```python
data6['K_Cluster'].value_counts()
```




    3    675
    0    643
    1    453
    2    452
    Name: K_Cluster, dtype: int64




```python
plt.figure(figsize=(14,8))
pl = sns.scatterplot(data = data6,x=data6["MntTotal"], y=data6["Income"],hue=data6["K_Cluster"], palette= 'cool')
plt.xlabel("Spent", size=16, color= 'white')
plt.ylabel("Income", size=16, color= 'white')
plt.title("KMean Cluster's Profile Based On Income And Spending", size=24, color= 'white')
plt.legend()

plt.figure(figsize=(14,8))
pl = sns.scatterplot(data = data6,x=data6["MntTotal"], y=data6["Income"],hue=data6["G_Cluster"], palette= 'cool')
plt.xlabel("Spent", size=16, color= 'white')
plt.ylabel("Income", size=16, color= 'white')
plt.title("GMM Cluster's Profile Based On Income And Spending", size=24, color= 'white')
plt.legend()
plt.show()

```


    
![png](output_40_0.png)
    



    
![png](output_40_1.png)
    



```python
plt.figure()
plt.figure(figsize=(12,8))
pl = sns.countplot(x=data6["TotalAcceptedCmp"],hue=data6["K_Cluster"], palette= 'cool')
plt.xlabel("Campaigns", size=12, color= 'white')
plt.ylabel("Count of Consumers", size=12, color= 'white')
pl.set_title("Count Of Total Campaigns Accepted by KMeans Clusters", size=12, color= 'white')
plt.legend(loc="upper right")

plt.figure(figsize=(6,4))
pl = sns.countplot(x=data6["TotalAcceptedCmp"],hue=data6["G_Cluster"], palette= 'cool')
plt.xlabel("Campaigns", size=12, color= 'white')
plt.ylabel("Count of Consumers", size=12, color= 'white')
pl.set_title("Count Of Total Campaigns Accepted by GMM Clusters", size=12, color= 'white')
plt.legend(loc="upper right")


plt.show()
```


    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_41_1.png)
    



    
![png](output_41_2.png)
    



```python
plt.figure()
plt.figure(figsize=(6,4))
pl=sns.boxenplot(y=data6["NumDealsPurchases"],x=data6["K_Cluster"], palette= 'cool')
pl.set_title("Number of Purchases during Deals by KMeans Cluster")
plt.figure(figsize=(6,4))
pl=sns.boxenplot(y=data6["NumDealsPurchases"],x=data6["G_Cluster"], palette= 'cool')
pl.set_title("Number of Purchases during Deals by GMM Cluster")
plt.show()
```


    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_42_1.png)
    



    
![png](output_42_2.png)
    



```python
data7 = data6
data7['HasPartner'] = data7["Marital_Status"].replace({'Single': '0', 
                                                      'Widow': '0',
                                                      'Divorced': '0',
                                                      'Together': '1',
                                                      'Married': '1'})
data7['HasPartner'] = data7['HasPartner'].astype('int')

data7['Education'] = data7["Education"].replace({'Below Bachelor': '0', 
                                                      'Bachelor': '1',
                                                      'Master': '2',
                                                      'PhD': '3'})
data7['Education'] = data7['Education'].astype('int')

Personal = ["CustomerFor", 'AvgWeb', "Age", "NumChildren", "HasPartner", 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'AverageCheck', "Education"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data7[i], y=data7["MntTotal"], hue =data7["K_Cluster"], kind="kde", palette='cool')
    plt.show()
```


    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_1.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_3.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_5.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_7.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_9.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_11.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_13.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_15.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_17.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_43_19.png)
    



```python

Personal = ["CustomerFor", 'AvgWeb', "Age", "NumChildren", "HasPartner", 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 'AverageCheck', "Education", "MntTotal"]

for i in Personal:
    plt.figure()
    sns.jointplot(x=data7[i], y=data7["Income"], hue =data7["K_Cluster"], kind="kde", palette='cool')
    plt.show()
```


    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_1.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_3.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_5.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_7.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_9.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_11.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_13.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_15.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_17.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_19.png)
    



    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_44_21.png)
    



```python
data6.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Age</th>
      <td>2223.0</td>
      <td>52.222222</td>
      <td>11.988485</td>
      <td>25.00</td>
      <td>44.000000</td>
      <td>51.000000</td>
      <td>62.000000</td>
      <td>128.000000</td>
    </tr>
    <tr>
      <th>CustomerFor</th>
      <td>2223.0</td>
      <td>1243.506973</td>
      <td>232.087024</td>
      <td>732.00</td>
      <td>1072.500000</td>
      <td>1245.000000</td>
      <td>1417.000000</td>
      <td>1795.000000</td>
    </tr>
    <tr>
      <th>Education</th>
      <td>2223.0</td>
      <td>1.663968</td>
      <td>0.839502</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>Income</th>
      <td>2223.0</td>
      <td>51885.707837</td>
      <td>20973.316223</td>
      <td>3502.00</td>
      <td>35533.500000</td>
      <td>51369.000000</td>
      <td>68277.500000</td>
      <td>162397.000000</td>
    </tr>
    <tr>
      <th>Kidhome</th>
      <td>2223.0</td>
      <td>0.443995</td>
      <td>0.538682</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>Teenhome</th>
      <td>2223.0</td>
      <td>0.506073</td>
      <td>0.544041</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>TotalEnrollYrs</th>
      <td>2223.0</td>
      <td>8.368471</td>
      <td>0.635829</td>
      <td>6.97</td>
      <td>7.900000</td>
      <td>8.370000</td>
      <td>8.840000</td>
      <td>9.880000</td>
    </tr>
    <tr>
      <th>Recency</th>
      <td>2223.0</td>
      <td>49.108862</td>
      <td>28.947372</td>
      <td>0.00</td>
      <td>24.000000</td>
      <td>49.000000</td>
      <td>74.000000</td>
      <td>99.000000</td>
    </tr>
    <tr>
      <th>MntWines</th>
      <td>2223.0</td>
      <td>305.250562</td>
      <td>337.066461</td>
      <td>0.00</td>
      <td>24.000000</td>
      <td>176.000000</td>
      <td>505.000000</td>
      <td>1493.000000</td>
    </tr>
    <tr>
      <th>MntFruits</th>
      <td>2223.0</td>
      <td>26.428250</td>
      <td>39.884105</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>33.000000</td>
      <td>199.000000</td>
    </tr>
    <tr>
      <th>MntMeatProducts</th>
      <td>2223.0</td>
      <td>164.159244</td>
      <td>215.102636</td>
      <td>0.00</td>
      <td>16.000000</td>
      <td>67.000000</td>
      <td>230.500000</td>
      <td>984.000000</td>
    </tr>
    <tr>
      <th>MntFishProducts</th>
      <td>2223.0</td>
      <td>37.589294</td>
      <td>54.581173</td>
      <td>0.00</td>
      <td>3.000000</td>
      <td>12.000000</td>
      <td>50.000000</td>
      <td>259.000000</td>
    </tr>
    <tr>
      <th>MntSweetProducts</th>
      <td>2223.0</td>
      <td>26.855151</td>
      <td>40.641920</td>
      <td>0.00</td>
      <td>1.000000</td>
      <td>8.000000</td>
      <td>33.000000</td>
      <td>198.000000</td>
    </tr>
    <tr>
      <th>MntGoldProds</th>
      <td>2223.0</td>
      <td>43.694107</td>
      <td>51.168585</td>
      <td>0.00</td>
      <td>9.000000</td>
      <td>24.000000</td>
      <td>56.000000</td>
      <td>262.000000</td>
    </tr>
    <tr>
      <th>NumDealsPurchases</th>
      <td>2223.0</td>
      <td>17.909163</td>
      <td>10.661608</td>
      <td>0.00</td>
      <td>7.690000</td>
      <td>16.670000</td>
      <td>25.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>NumWebPurchases</th>
      <td>2223.0</td>
      <td>26.664098</td>
      <td>9.445886</td>
      <td>0.00</td>
      <td>20.000000</td>
      <td>26.090000</td>
      <td>33.330000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>NumCatalogPurchases</th>
      <td>2223.0</td>
      <td>14.128709</td>
      <td>12.487030</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>12.000000</td>
      <td>22.220000</td>
      <td>61.110000</td>
    </tr>
    <tr>
      <th>NumStorePurchases</th>
      <td>2223.0</td>
      <td>41.117512</td>
      <td>11.704320</td>
      <td>0.00</td>
      <td>33.330000</td>
      <td>41.180000</td>
      <td>50.000000</td>
      <td>100.000000</td>
    </tr>
    <tr>
      <th>NumWebVisitsMonth</th>
      <td>2223.0</td>
      <td>5.313540</td>
      <td>2.359505</td>
      <td>0.00</td>
      <td>3.000000</td>
      <td>6.000000</td>
      <td>7.000000</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>AcceptedCmp3</th>
      <td>2223.0</td>
      <td>0.073324</td>
      <td>0.260727</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>AcceptedCmp4</th>
      <td>2223.0</td>
      <td>0.074674</td>
      <td>0.262924</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>AcceptedCmp5</th>
      <td>2223.0</td>
      <td>0.073324</td>
      <td>0.260727</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>AcceptedCmp1</th>
      <td>2223.0</td>
      <td>0.064777</td>
      <td>0.246188</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>AcceptedCmp2</th>
      <td>2223.0</td>
      <td>0.013495</td>
      <td>0.115409</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Complain</th>
      <td>2223.0</td>
      <td>0.009447</td>
      <td>0.096756</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Response</th>
      <td>2223.0</td>
      <td>0.150247</td>
      <td>0.357394</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>mnt_products</th>
      <td>2223.0</td>
      <td>603.976608</td>
      <td>601.544911</td>
      <td>5.00</td>
      <td>68.000000</td>
      <td>395.000000</td>
      <td>1042.500000</td>
      <td>2525.000000</td>
    </tr>
    <tr>
      <th>NumChildren</th>
      <td>2223.0</td>
      <td>0.950067</td>
      <td>0.751074</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>MntTotal</th>
      <td>2223.0</td>
      <td>603.976608</td>
      <td>601.544911</td>
      <td>5.00</td>
      <td>68.000000</td>
      <td>395.000000</td>
      <td>1042.500000</td>
      <td>2525.000000</td>
    </tr>
    <tr>
      <th>AvgWeb</th>
      <td>2223.0</td>
      <td>1.057472</td>
      <td>0.954960</td>
      <td>0.00</td>
      <td>0.330000</td>
      <td>0.750000</td>
      <td>1.500000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>Expenses</th>
      <td>2223.0</td>
      <td>603.976608</td>
      <td>601.544911</td>
      <td>5.00</td>
      <td>68.000000</td>
      <td>395.000000</td>
      <td>1042.500000</td>
      <td>2525.000000</td>
    </tr>
    <tr>
      <th>TotalAcceptedCmp</th>
      <td>2223.0</td>
      <td>0.449843</td>
      <td>0.893057</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>NumTotalPurchases</th>
      <td>2223.0</td>
      <td>99.819483</td>
      <td>4.239012</td>
      <td>0.00</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>100.010000</td>
    </tr>
    <tr>
      <th>ExpensePer</th>
      <td>2223.0</td>
      <td>0.951997</td>
      <td>0.776039</td>
      <td>0.00</td>
      <td>0.230000</td>
      <td>0.790000</td>
      <td>1.545000</td>
      <td>5.690000</td>
    </tr>
    <tr>
      <th>NumAllPurchases</th>
      <td>2223.0</td>
      <td>81.910319</td>
      <td>11.190025</td>
      <td>0.00</td>
      <td>75.000000</td>
      <td>83.330000</td>
      <td>92.300000</td>
      <td>100.010000</td>
    </tr>
    <tr>
      <th>AverageCheck</th>
      <td>2223.0</td>
      <td>6.852186</td>
      <td>6.441847</td>
      <td>0.10</td>
      <td>0.900000</td>
      <td>4.800000</td>
      <td>11.500000</td>
      <td>26.700000</td>
    </tr>
    <tr>
      <th>percentWines</th>
      <td>2223.0</td>
      <td>46.037770</td>
      <td>22.736823</td>
      <td>0.00</td>
      <td>29.158396</td>
      <td>45.833333</td>
      <td>64.116379</td>
      <td>96.330275</td>
    </tr>
    <tr>
      <th>percentMeat</th>
      <td>2223.0</td>
      <td>24.834430</td>
      <td>12.181276</td>
      <td>0.00</td>
      <td>15.645385</td>
      <td>23.333333</td>
      <td>32.758621</td>
      <td>74.908425</td>
    </tr>
    <tr>
      <th>percentFruits</th>
      <td>2223.0</td>
      <td>4.961564</td>
      <td>5.582569</td>
      <td>0.00</td>
      <td>0.900595</td>
      <td>3.007519</td>
      <td>7.043735</td>
      <td>44.554455</td>
    </tr>
    <tr>
      <th>percentFish</th>
      <td>2223.0</td>
      <td>7.170121</td>
      <td>7.800973</td>
      <td>0.00</td>
      <td>1.262245</td>
      <td>4.828273</td>
      <td>10.456439</td>
      <td>59.090909</td>
    </tr>
    <tr>
      <th>percentSweets</th>
      <td>2223.0</td>
      <td>5.024363</td>
      <td>5.773879</td>
      <td>0.00</td>
      <td>0.864045</td>
      <td>3.333333</td>
      <td>7.022681</td>
      <td>60.000000</td>
    </tr>
    <tr>
      <th>percentGold</th>
      <td>2223.0</td>
      <td>11.971752</td>
      <td>10.683095</td>
      <td>0.00</td>
      <td>3.815706</td>
      <td>8.571429</td>
      <td>16.978220</td>
      <td>70.241287</td>
    </tr>
    <tr>
      <th>K_Cluster</th>
      <td>2223.0</td>
      <td>1.521368</td>
      <td>1.198321</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>G_Cluster</th>
      <td>2223.0</td>
      <td>1.257760</td>
      <td>0.966892</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>HasPartner</th>
      <td>2223.0</td>
      <td>0.644175</td>
      <td>0.478870</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
g = sns.catplot(x="K_Cluster", y="Income",
                hue="HasChildren", col="HasAPartner",
                data=data6, kind="violin", split=True, palette = 'cool')
g.set_axis_labels("Cluster", "Income",size=14, color = 'White')
g.set_titles(col_template="{col_name} Partner", row_template="{row_name}", size=16, color = 'White')
```




    <seaborn.axisgrid.FacetGrid at 0x1a6cbf86340>




    
![png](output_46_1.png)
    



```python
plt.figure()
plt.figure(figsize=(6,4))
pl=sns.boxenplot(y=data6["Education"],x=data6["K_Cluster"], palette= 'cool')
pl.set_title("Number of Purchases during Deals by KMeans Cluster")
```




    Text(0.5, 1.0, 'Number of Purchases during Deals by KMeans Cluster')




    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_47_2.png)
    



```python
data6['Education'] = data['Education'].astype('category')
data6['Education'].replace(['Basic','2n Cycle', 'Graduation'], 
                          ['Below Bachelor', 'Master', 'Bachelor'], inplace=True)
data6
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>CustomerFor</th>
      <th>Education</th>
      <th>Marital_Status</th>
      <th>Income</th>
      <th>Kidhome</th>
      <th>Teenhome</th>
      <th>TotalEnrollYrs</th>
      <th>Recency</th>
      <th>MntWines</th>
      <th>...</th>
      <th>AverageCheck</th>
      <th>percentWines</th>
      <th>percentMeat</th>
      <th>percentFruits</th>
      <th>percentFish</th>
      <th>percentSweets</th>
      <th>percentGold</th>
      <th>K_Cluster</th>
      <th>G_Cluster</th>
      <th>HasPartner</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>64</td>
      <td>1703</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>58138.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.63</td>
      <td>58</td>
      <td>635</td>
      <td>...</td>
      <td>18.4</td>
      <td>39.270254</td>
      <td>33.766234</td>
      <td>5.442177</td>
      <td>10.636982</td>
      <td>5.442177</td>
      <td>5.442177</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67</td>
      <td>857</td>
      <td>Bachelor</td>
      <td>Single</td>
      <td>46344.0</td>
      <td>1</td>
      <td>1</td>
      <td>7.31</td>
      <td>38</td>
      <td>11</td>
      <td>...</td>
      <td>0.4</td>
      <td>40.740741</td>
      <td>22.222222</td>
      <td>3.703704</td>
      <td>7.407407</td>
      <td>3.703704</td>
      <td>22.222222</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>56</td>
      <td>1204</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>71613.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.26</td>
      <td>26</td>
      <td>426</td>
      <td>...</td>
      <td>8.1</td>
      <td>54.896907</td>
      <td>16.365979</td>
      <td>6.314433</td>
      <td>14.304124</td>
      <td>2.706186</td>
      <td>5.412371</td>
      <td>2</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>37</td>
      <td>797</td>
      <td>Bachelor</td>
      <td>Together</td>
      <td>26646.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.15</td>
      <td>26</td>
      <td>11</td>
      <td>...</td>
      <td>0.7</td>
      <td>20.754717</td>
      <td>37.735849</td>
      <td>7.547170</td>
      <td>18.867925</td>
      <td>5.660377</td>
      <td>9.433962</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>40</td>
      <td>1053</td>
      <td>PhD</td>
      <td>Married</td>
      <td>58293.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.85</td>
      <td>94</td>
      <td>173</td>
      <td>...</td>
      <td>5.7</td>
      <td>40.995261</td>
      <td>27.962085</td>
      <td>10.189573</td>
      <td>10.900474</td>
      <td>6.398104</td>
      <td>3.554502</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2235</th>
      <td>54</td>
      <td>1273</td>
      <td>Bachelor</td>
      <td>Married</td>
      <td>61223.0</td>
      <td>0</td>
      <td>1</td>
      <td>8.45</td>
      <td>46</td>
      <td>709</td>
      <td>...</td>
      <td>15.1</td>
      <td>52.870992</td>
      <td>13.571961</td>
      <td>3.206562</td>
      <td>3.131991</td>
      <td>8.799403</td>
      <td>18.419090</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2236</th>
      <td>75</td>
      <td>793</td>
      <td>PhD</td>
      <td>Together</td>
      <td>64014.0</td>
      <td>2</td>
      <td>1</td>
      <td>7.13</td>
      <td>56</td>
      <td>406</td>
      <td>...</td>
      <td>6.5</td>
      <td>91.441441</td>
      <td>6.756757</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.801802</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2237</th>
      <td>40</td>
      <td>1047</td>
      <td>Bachelor</td>
      <td>Divorced</td>
      <td>56981.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.83</td>
      <td>91</td>
      <td>908</td>
      <td>...</td>
      <td>13.1</td>
      <td>73.166801</td>
      <td>17.485898</td>
      <td>3.867849</td>
      <td>2.578566</td>
      <td>0.966962</td>
      <td>1.933924</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2238</th>
      <td>65</td>
      <td>1048</td>
      <td>Master</td>
      <td>Together</td>
      <td>69245.0</td>
      <td>0</td>
      <td>1</td>
      <td>7.83</td>
      <td>8</td>
      <td>428</td>
      <td>...</td>
      <td>9.2</td>
      <td>50.771056</td>
      <td>25.385528</td>
      <td>3.558719</td>
      <td>9.489917</td>
      <td>3.558719</td>
      <td>7.236062</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2239</th>
      <td>67</td>
      <td>1514</td>
      <td>PhD</td>
      <td>Married</td>
      <td>52869.0</td>
      <td>1</td>
      <td>1</td>
      <td>9.11</td>
      <td>40</td>
      <td>84</td>
      <td>...</td>
      <td>2.4</td>
      <td>48.837209</td>
      <td>35.465116</td>
      <td>1.744186</td>
      <td>1.162791</td>
      <td>0.581395</td>
      <td>12.209302</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2223 rows × 50 columns</p>
</div>




```python
plt.figure()
plt.figure(figsize=(12,8))
pl = sns.countplot(x=data6["Education"],hue=data6["K_Cluster"], palette= 'cool')
plt.xlabel("Education", size=12, color= 'white')
plt.ylabel("Count of Consumers", size=12, color= 'white')
pl.set_title("Count Of Education by Cluster", size=12, color= 'white')
plt.legend(loc="upper right")
```




    <matplotlib.legend.Legend at 0x1a6d32dbee0>




    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_49_2.png)
    



```python
plt.figure()
plt.figure(figsize=(12,8))
pl = sns.countplot(x=data6["Marital_Status"],hue=data6["K_Cluster"], palette= 'cool')
plt.xlabel("Relationship", size=12, color= 'white')
plt.ylabel("Count of Consumers", size=12, color= 'white')
pl.set_title("Relationship Status by Cluster", size=12, color= 'white')
plt.legend(loc="upper right")
```




    <matplotlib.legend.Legend at 0x1a6d3751b20>




    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_50_2.png)
    



```python
plt.figure()
plt.figure(figsize=(12,8))
pl = sns.countplot(x=data6["NumChildren"],hue=data6["K_Cluster"], palette= 'cool')
plt.xlabel("Number of Children", size=12, color= 'white')
plt.ylabel("Count of Consumers", size=12, color= 'white')
pl.set_title("Amount of Children by Cluster", size=12, color= 'white')
plt.legend(loc="upper right")
```




    <matplotlib.legend.Legend at 0x1a6ba5c7a30>




    <Figure size 345.6x259.2 with 0 Axes>



    
![png](output_51_2.png)
    



```python
plt.figure(figsize=(14,8))
pl = sns.scatterplot(data = data6,x=data6["Age"], y=data6["Income"],hue=data6["K_Cluster"], palette= 'cool')
plt.xlabel("Age", size=16, color= 'white')
plt.ylabel("Income", size=16, color= 'white')
plt.title("KMean Cluster's Profile Based On Income And Spending", size=24, color= 'white')
plt.legend()
```




    <matplotlib.legend.Legend at 0x1a6ce0d5880>




    
![png](output_52_1.png)
    



```python

```
