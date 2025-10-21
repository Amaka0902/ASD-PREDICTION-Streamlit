# Check what unique values are in your target column
print("Unique values in target column before cleaning:", data['result'].unique())

# Drop rows where target is missing
data = data.dropna(subset=['result'])

# Confirm again
print("Unique values in target column after dropping NaN:", data['result'].unique())

# Separate features and target
X = data.drop(columns=['result'])
y = data['result']

print("Number of rows in dataset:", len(data))
print("Distribution of target values:\n", y.value_counts())
