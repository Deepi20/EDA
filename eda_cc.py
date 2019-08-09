import pandas as pd
path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/auto.csv"
df = pd.read_csv(path,header = None)
df.head(5)
df.tail(10)
headers = ["symboling","normalized-losses","make","fuel-type","aspiration", "num-of-doors","body-style",
         "drive-wheels","engine-location","wheel-base", "length","width","height","curb-weight","engine-type",
         "num-of-cylinders", "engine-size","fuel-system","bore","stroke","compression-ratio","horsepower",
         "peak-rpm","city-mpg","highway-mpg","price"]
print("headers\n",headers)
df.columns = headers
df.head(10)
df.dropna(subset = ["price"],axis = 0)
print(df.columns)
df.to_csv("automobile.csv",index= False)
df.dtypes
print(df.dtypes)
df.describe()
df.describe(include = "all")
df[['length','compression-ratio']].describe()
df.info
import matplotlib.pylab as plt
import numpy as np
df.replace("?",np.nan,inplace = True)
df.head(5)
missing_data = df.isnull()
missing_data.head(5)
for column in missing_data.columns.values.tolist():
    print(column)
    print(missing_data[column].value_counts())
    print("")
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis = 0)
print("Average of normalized-losses:",avg_norm_loss)
df["normalized-losses"].replace(np.nan,avg_norm_loss,inplace = True)
avg_bore = df["bore"].astype("float").mean(axis = 0)
print("Avg of bore:",avg_bore)
df["bore"].replace(np.nan,avg_bore,inplace = True)
avg_stroke = df["stroke"].astype("float").mean(axis = 0)
print("Avg of stroke:",avg_stroke)
df["stroke"].replace(np.nan,avg_stroke,inplace = True)
avg_horsepower = df['horsepower'].astype('float').mean(axis=0)
print("Average horsepower:", avg_horsepower)
df['horsepower'].replace(np.nan, avg_horsepower, inplace=True)
avg_peakrpm=df['peak-rpm'].astype('float').mean(axis=0)
print("Average peak rpm:", avg_peakrpm)
df['peak-rpm'].replace(np.nan, avg_peakrpm, inplace=True)
df['num-of-doors'].value_counts()
df['num-of-doors'].value_counts().idxmax()
df["num-of-doors"].replace(np.nan, "four", inplace=True)
df.dropna(subset = ["price"],axis = 0,inplace = True)
df.reset_index(drop = True, inplace = True)
df.head(5)
df.isnull()
df.dtypes
df[["bore","stroke"]]=df[["bore","stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")
df.head(10)
df["city-L/100km"] = 235/df["city-mpg"]
df.head(10)
df["highway-L/100km"] = 235/df["highway-mpg"]
df.head(5)
df.shape
df = df.drop(['city-mpg','highway-mpg'],axis=1)
df.columns
df["length"] = df["length"]/df["length"].max()
df["width"] = df["width"]/df["width"].max()
df["height"] = df["height"]/df["height"].max()
df["horsepower"] = df["horsepower"].astype(int, copy = True)
%matplotlib inline
import matplotlib as plt
from matplotlib import pyplot
plt.pyplot.hist(df["horsepower"])
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
bins = np.linspace(min(df["horsepower"]),max(df["horsepower"]),4)
bins
group_names = ["Low","Medium","High"]
df["horsepower-binned"] = pd.cut(df["horsepower"],bins,labels = group_names,include_lowest = True)
df[["horsepower","horsepower-binned"]].head(30)
df["horsepower-binned"].value_counts()
pyplot.bar(group_names, df["horsepower-binned"].value_counts())
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
a = (0,1,2)
plt.pyplot.hist(df["horsepower"],bins = 3)
plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
df.columns
dummy_variable_1 = pd.get_dummies(df["fuel-type"])
dummy_variable_1.head()
dummy_variable_1.rename(columns = {'fuel-type-diesel':'diesel','fuel-type-gas':'gas'},inplace = True)
dummy_variable_1.head()
df = pd.concat([df,dummy_variable_1],axis = 1)
df.drop(["fuel-type"],axis = 1,inplace = True)
df.head(5)
dummy_variable_2 = pd.get_dummies(df["aspiration"])
dummy_variable_2.head()
dummy_variable_2.rename(columns = {'aspiration-std':'std','aspiration-turbo':'turbo'}, inplace = True )
dummy_variable_2.head()
df = pd.concat([df,dummy_variable_2],axis = 1)

df.head(5)
df.to_csv('clean_df.csv')
df.corr()
sns.regplot(x="engine-size", y="price", data=df)
plt.ylim(0,)
df[["engine-size", "price"]].corr()
sns.regplot(x="highway-mpg", y="price", data=df)
df[['highway-mpg', 'price']].corr()
sns.regplot(x="peak-rpm", y="price", data=df)
df[['peak-rpm','price']].corr()
sns.boxplot(x="body-style", y="price", data=df)
sns.boxplot(x="engine-location", y="price", data=df)
sns.boxplot(x="drive-wheels", y="price", data=df)
df.describe()
df['drive-wheels'].value_counts()
df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts = df['drive-wheels'].value_counts().to_frame()
drive_wheels_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
drive_wheels_counts
drive_wheels_counts.index.name = 'drive-wheels'
drive_wheels_counts
engine_loc_counts = df['engine-location'].value_counts().to_frame()
engine_loc_counts.rename(columns={'engine-location': 'value_counts'}, inplace=True)
engine_loc_counts.index.name = 'engine-location'
engine_loc_counts.head(10)
df['drive-wheels'].unique()
df_group_one = df[['drive-wheels','body-style','price']]
df_group_one = df_group_one.groupby(['drive-wheels'],as_index=False).mean()
df_group_one
df_gptest = df[['drive-wheels','body-style','price']]
grouped_test1 = df_gptest.groupby(['drive-wheels','body-style'],as_index=False).mean()
grouped_test1
