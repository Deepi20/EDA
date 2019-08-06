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
