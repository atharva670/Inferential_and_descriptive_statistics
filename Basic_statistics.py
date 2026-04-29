import pandas as pd
import numpy as np
import math

# Data cleaning

data = pd.read_csv('train.csv')
df = pd.DataFrame(data)
# mean imputation
df['Item_Weight'] = pd.to_numeric(df['Item_Weight'], errors='coerce')
df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
# mode imputation
mod = df['Outlet_Size'].mode().iloc[0]
print("Mode=", mod)
df['Outlet_Size'] = df['Outlet_Size'].apply(
    lambda x: np.nan if (str(x).isdigit() or x == '') else x
)
df['Outlet_Size'] = df['Outlet_Size'].fillna('Medium')
# Checking for duplicate rows
print(df.duplicated().sum())
# Outlier Detection and Removal
columns=df.select_dtypes(include=['int64','float64']).columns
for i in columns:
    Q1=df[i].quantile(0.25)
    Q3=df[i].quantile(0.75)
    IQR=Q3-Q1
    low=Q1-(1.5*IQR)
    high=Q3+(1.5*IQR)
    sum=0
    for j in df[i]:
        if(j<low or j>high):
            sum=sum+1
    print(i,"Outlier=",sum)


df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].mean())
df.to_csv('train.csv', index=False)
# Descriptive statistics
df1 = df[df['Outlet_Type'] == 'Supermarket Type1']
val = df1['Item_Outlet_Sales'].mean()
print("Average Sales from Supermarket Type1 is ", val, 'INR')
df2 = df[df['Outlet_Type'] == 'Supermarket Type2']
val2 = df2['Item_Outlet_Sales'].mean()
print("Average Sales from Supermarket Type2 is ", val2, 'INR')
df3 = df[df['Outlet_Type'] == 'Supermarket Type3']
val3 = df3['Item_Outlet_Sales'].mean()
print("Average Sales from Supermarket Type3 is ", val3, 'INR')
df4 = df[df['Outlet_Type'] == 'Grocery Store']
val4 = df4['Item_Outlet_Sales'].mean()
print("Average Sales from Grocery Store is ", val4, 'INR')
print("Maximum People  go in Supermarket Type 3")
l = []
for i in df3['Item_MRP']:
    l.append(i)
l.sort()

print(
    "Minimum expense that one person can do in Supermarket Type 3 is ",
    l[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Supermarket Type 3 is ",
    l[len(l) - 1],
    'INR',
)
mod = df['Outlet_Location_Type'].mode().iloc[0]
print("Most of the Outlets are in", mod, "Cities")
print(
    "Difference between minimum expense and maximum expense in Supermarket 3 is",
    l[len(l) - 1] - l[0],
)
# Standard Deviation and Variance is used for machine learning models
l2 = []
for i in l:
    if i < np.median(l):
        l2.append(i)

l3 = []
for i in l:
    if i > np.median(l):
        l3.append(i)
print(
    "25% of products in Supermarket 3  having cost less than or equal to",
    np.median(l2),
)
print(
    "Half of the  products in Supermarket 3 having cost less than or equal to",
    np.median(l),
)
print(
    "75% of products  in Supermarket 3 having cost less than or equal to",
    np.median(l3),
)
print(
    "25% of products in Supermarket 3 having cost greater than", np.median(l3)
)
# SuperMarket 2
l4 = []
for i in df2['Item_MRP']:
    l4.append(i)
l4.sort()

print(
    "Minimum expense that one person can do in Supermarket Type 2 is ",
    l4[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Supermarket Type 2 is ",
    l4[len(l4) - 1],
    'INR',
)
print(
    "Difference between minimum expense and maximum expense in Supermarket 2 is",
    l4[len(l4) - 1] - l4[0],
)
# Standard Deviation and Variance is used for machine learning models
l5 = []
for i in l:
    if i < np.median(l4):
        l5.append(i)

l6 = []
for i in l4:
    if i > np.median(l4):
        l6.append(i)
print(
    "25% of products in Supermarket 2 having cost less than or equal to",
    np.median(l5),
)
print(
    "Half of the  products in Supermarket 2 having cost less than or equal to",
    np.median(l4),
)
print(
    "75% of products  in Supermarket 2 having cost less than or equal to",
    np.median(l6),
)
print(
    "25% of products in Supermarket 2 having cost greater than", np.median(l6)
)
# SuperMarket 1
l7 = []
for i in df1['Item_MRP']:
    l7.append(i)
l7.sort()

print(
    "Minimum expense that one person can do in Supermarket Type 1 is ",
    l7[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Supermarket Type 1 is ",
    l7[len(l7) - 1],
    'INR',
)
print(
    "Difference between minimum expense and maximum expense in Supermarket 1 is",
    l7[len(l7) - 1] - l7[0],
)

l8 = []
for i in l7:
    if i < np.median(l7):
        l8.append(i)

l9 = []
for i in l7:
    if i > np.median(l7):
        l9.append(i)
print(
    "25% of products in Supermarket 1 having cost less than or equal to",
    np.median(l8),
)
print(
    "Half of the  products in Supermarket 1 having cost less than or equal to",
    np.median(l7),
)
print(
    "75% of products  in Supermarket 1 having cost less than or equal to",
    np.median(l9),
)
print(
    "25% of products in Supermarket 1 having cost greater than", np.median(l9)
)
# Grocery Store
l10 = []
for i in df4['Item_MRP']:
    l10.append(i)
l10.sort()
print(
    "Minimum expense that one person can do in Grocery Store is ",
    l10[0],
    'INR',
)
print(
    "Maximum expense that one person can do in Grocery Store is ",
    l10[len(l10) - 1],
    'INR',
)
print(
    "Difference between minimum expense and maximum expense in Grocery Store is",
    l10[len(l10) - 1] - l10[0],
)
# Standard Deviation and Variance is used for machine learning models
l11 = []
for i in l10:
    if i < np.median(l10):
        l11.append(i)

l12 = []
for i in l10:
    if i > np.median(l10):
        l12.append(i)
print(
    "25% of products in Grocery Store  having cost less than or equal to",
    np.median(l11),
)
print(
    "Half of the  products in Grocery Store having cost less than or equal to",
    np.median(l10),
)
print(
    "75% of products  in Grocery Store having cost less than or equal to",
    np.median(l12),
)
print(
    "25% of products in Grocery Store having cost greater than", np.median(l12)
)
print(
    "For  maximum shopping with minimum cost People should go to Supermarket 3"
)
# Standard Deviation and Variance is used for machine learning models
# Inferential statistics
# Hypothesis Testing

print(
    "Let H0=Half of the  products in Supermarket 3 having cost less than or equal to",
    np.median(l),
)
print(
    "Let H1=Half of the  products in Supermarket 3 is not having cost less than or equal to",
    np.median(l),
)
n = 935 / 2
num = int(n)

sum = 0
for i in range(num):

    sum = sum + df3['Item_MRP'].iloc[i]


X = sum / num  # Sample mean
print("Sample mean", X)
Y = df3['Item_MRP'].mean()  # Population mean
print("Population mean", Y)
sum1 = 0
for i in range(len(df3)):
    sum1 = sum1 + (
        (df3['Item_MRP'].iloc[i] - Y) * (df3['Item_MRP'].iloc[i] - Y)
    )
sd = math.sqrt(sum1 / len(df3))
print("Standard Deviation=", sd)
sample_size = len(df3)
print("Sample size=", sample_size)
d = sd / math.sqrt(sample_size)
Z = (X - Y) / d
print("Z-Value=", Z)
print("Since Z-value is near to Zero , p-value is very High")
print("Fail to reject H0")
