import pandas as pd
customers = pd.read_csv("data/raw/customers.csv")
print(customers.head())
customers.info()
print(customers.isnull().sum())
print(customers.duplicated().sum())

customers["city"] = customers["city"].str.strip().str.title()
print(customers["city"].unique())

customers["gender"] = customers["gender"].str.strip().str.lower()
customers["gender"] = customers["gender"].replace({"m": "male", "f" : "female"})
customers["gender"] = customers["gender"].fillna("unknown")
print(customers["gender"].unique())

customers["age"] = pd.to_numeric(customers["age"], errors = "coerce")
customers.loc[~customers["age"].between(10, 100), "age"] = float("nan")
customers["age"] = customers["age"].fillna(customers.groupby("city")["age"].transform("median"))

customers["signup_date"] = pd.to_datetime(customers["signup_date"], errors = "coerce")
print(customers["signup_date"].dtype)

customers.info()

customers.to_csv("data/clean/customers.csv", index = False)

