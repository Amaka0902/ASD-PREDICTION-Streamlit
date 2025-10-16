#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd 

#Load dataset 
df = pd.read_csv("test.csv")

# Displaying intial info 
print ("Initial shape:",df.shape)
print("n\Missing values per column:\n") , df.isnull().sum()

# Replace "?" with NaN
df.replace("?", pd.NA, inplace=True)

# Fill missing values: 
# For numeric columns, keep as is; for object columns, fill with "N/A"
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna("N/A", inplace=True)
    else:
        # For numeric columns, we can either fill with mean or leave as NaN
        df[col] = df[col].astype(float)  # ensures numeric type
        df[col].fillna(df[col], inplace=True)  # keeps NaN intact for numerics

# Remove duplicates
df.drop_duplicates(inplace=True)

# Display cleaned dataset info
print("\nAfter cleaning:")
print("Shape:", df.shape)
print("Remaining missing values:\n", df.isnull().sum())

# Save cleaned dataset
df.to_csv("test_cleaned.csv", index=False)

print("\nCleaned dataset saved as 'test_cleaned.csv'")


# In[ ]:




