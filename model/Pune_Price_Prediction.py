#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns   


# In[2]:


house_df = pd.read_csv("model/Pune_House_Data.csv")
house_df


# In[3]:


house_df.shape


# In[4]:


house_df.info()


# In[5]:


house_df['area_type'].value_counts()


# In[6]:


sns.heatmap(house_df.isnull())


# In[7]:


house_df.isnull().sum()


# In[ ]:





# In[8]:


plt.figure(figsize=(15,7)) 
sns.distplot(house_df['price']) 
plt.show()


# In[9]:


sns.pairplot(house_df, diag_kind='kde')


# In[10]:


house_df.corr()


# In[11]:


sns.heatmap(house_df.corr(), annot=True)


# In[12]:


# Filling Misssing Values
# for all object type missing values, impute mode of corresponding column
for column in ['site_location', 'size']:
    house_df[column].fillna(house_df[column].mode()[0], inplace = True)


# In[13]:


# fill numerical values
#plt.figure(figsize=(5,5))
sns.boxplot(house_df.bath)
plt.show()


# In[14]:


# bath column has outliers, treat with median value imputation
house_df['bath'].fillna(house_df['bath'].median(), inplace = True)


# In[15]:


sns.boxplot(house_df.balcony)
plt.show()


# In[16]:


# no outiers in balcony, so treat missing values with mean value imputation
house_df['balcony'].fillna(house_df['balcony'].mean(), inplace = True)


# In[17]:


house_df.drop(['society', 'area_type','availability',"balcony"], axis=1, inplace=True)


# In[18]:


house_df.isnull().sum()


# In[19]:


plt.figure(figsize=(15,7)) 
sns.distplot(house_df['price']) 
plt.show()


# In[20]:


sns.pairplot(house_df, diag_kind='kde')


# In[21]:


house_df['bhk_size'] = house_df['size'].apply(lambda x: int(x.split()[0]))
house_df.drop('size', axis = 1, inplace = True)
# house_df.groupby('bhk_size')['bhk_size'].agg('count')
house_df['bhk_size'].value_counts()


# In[22]:


house_df.head(3)


# In[23]:


house_df['total_sqft'].unique()


# In[24]:


# Since the total_sqft contains range values such as 1133-1384, lets filter out these values
def isFloat(x):
    try:
        float(x)
    except:
        return False
    return True


# In[25]:


# Displaying all the rows that are not integers
house_df[~house_df['total_sqft'].apply(isFloat)]


# In[26]:


# Converting the range values to integer values and removing other types of error
def convert_sqft_to_num(x):
    tokens = x.split('-')
    if len(tokens) == 2:
        return (float(tokens[0])+float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[27]:


house_df['new_total_sqft'] = house_df['total_sqft'].apply(convert_sqft_to_num)
house_df = house_df.drop('total_sqft', axis=1)
house_df


# In[28]:


house_df.dtypes


# In[29]:


# house_df1 = house_df.copy()
# for index in new_df.index:
#     if index in house_df1.index:
#         house_df1.loc[index, 'total_sqft'] = new_df.loc[index, 'total_sqft']
# house_df1.head()


# In[30]:


print(house_df['new_total_sqft'].unique())


# In[31]:


locations_count = house_df['site_location'].value_counts(ascending=True)
locations_count


# In[32]:


# scaling
# change price in lacs to price_per_sqft
house_df1 = house_df.copy()

# In our dataset the price column is in Lakhs
# price in lakhs = total_sqft * price_per_sqft
# thus price_per_sqft = price in lakhs/total_sqft

house_df1['price_per_sqft'] = (house_df1['price']*100000)/house_df1['new_total_sqft']
#house_df1.drop('price', axis = 1, inplace = True)
house_df1.head()


# ## Remove Outliers 
As a data scientist when you have a conversation with your business manager (who has expertise 
in real estate), he will tell you that normally square ft per bedroom is 300 
(i.e. 2 bhk apartment is minimum 600 sqft. If you have for example 400 sqft apartment with 2 bhk
than that seems suspicious and can be removed as an outlier. 
We will remove such outliers by keeping our minimum thresold per bhk to be 300 sqft
# In[33]:


# Removing the rows that have 1 Room for less than 300sqft
house_df1[house_df1.new_total_sqft/house_df1.bhk_size<300].head()

Check above data points. We have 6 bhk apartment with 1020 sqft. 
Another one is 8 bhk and total sqft is 600. 
These are clear data errors that can be removed safely
# In[34]:


df = house_df1[~(house_df1.new_total_sqft/house_df1.bhk_size<300)]
df.shape


# In[35]:


sns.boxplot(df['price'])


# In[36]:


#Outlier Removal Using Standard Deviation and Mean
df.price_per_sqft.describe()

Here we find that min price per sqft is 267 rs/sqft whereas max is 12000000, this shows a wide variation in property prices. We should remove outliers per location using mean and one standard deviation
# In[37]:


def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('site_location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df2 = remove_pps_outliers(df)
df2.shape


# In[38]:


df.columns


# In[39]:


def plot_scatter_chart(df,location):
    bhk2 = df[(df.site_location==location) & (df.bhk_size==2)]
    bhk3 = df[(df.site_location==location) & (df.bhk_size==3)]
    plt.rcParams['figure.figsize'] = (15,10)
    plt.scatter(bhk2.new_total_sqft, bhk2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(bhk3.new_total_sqft, bhk3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df2,"Bhavani Peth")

We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per bhk, i.e.
# In[40]:


#Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft 
#of 1 BHK apartment

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('site_location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk_size'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk_size'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                #print(bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df3 = remove_bhk_outliers(df2)
# df8 = df7.copy()
df3.shape


# In[41]:


plot_scatter_chart(df3,"Bhavani Peth")


# In[42]:


Outlier Removal Using Bathrooms Feature


# In[43]:


plt.hist(df3.bath,rwidth=0.8)
plt.xlabel("Number of bathrooms")
plt.ylabel("Count")


# In[44]:


sns.boxplot(house_df['bath'])


# In[45]:


df3[df3.bath>10]


# In[46]:


sns.boxplot(house_df['bhk_size'])


# In[47]:


#It is unusual to have 2 more bathrooms than number of bedrooms in a home
df3[df3.bath > df3.bhk_size+2]


# In[48]:


# the business manager has a conversation with you (i.e. a data scientist) that if you have
#4 bedroom home and even if you have bathroom in all 4 rooms plus one guest bathroom, 
#you will have total bath = total bed + 1 max. Anything above that is an outlier or a data error and can be removed

df3 = df3[df3.bath < df3.bhk_size+2]
df3.shape


# In[49]:


df3.drop(['price_per_sqft'], axis='columns', inplace=True)
df3.head()


# In[50]:


df = df3.copy()


# In[51]:


dummy_cols = pd.get_dummies(df.site_location)
df = pd.concat([df,dummy_cols], axis='columns')


# In[52]:


df.drop(['site_location'], axis='columns', inplace=True)
df.head()


# In[53]:


df.shape


# In[54]:


df.isna().sum()


# In[55]:


# Splitting the dataset into features and label
x = df.drop('price', axis='columns')
y = df['price']


# In[56]:


# Using GridSearchCV to find the best algorithm for this problem
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import LinearRegression


# In[57]:


# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=20)


# In[58]:


# Creating Linear Regression Model
from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)
model.fit(x_train, y_train)


# In[59]:


model.score(x_test, y_test)


# In[ ]:





# In[60]:


import pickle
with open ('pune_house_price_model.pickle', 'wb') as f:
    pickle.dump(model, f)


# In[61]:


# Export location and column information to a file that will be useful later 
#on in our prediction application

import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("pune_columns.json","w") as f:
    f.write(json.dumps(columns))


# In[62]:


x.columns


# In[ ]:





# In[ ]:




