import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from pycaret.classification import (
    setup,
    compare_models,
    create_model,
    tune_model,
    evaluate_model,
    predict_model,
    save_model,
    load_model
)

data = pd.read_csv("dataset.csv", na_values=["NaN", "NULL", "?", " "])


data = data.drop_duplicates()


for col in data.columns:
    if data[col].dtype in ["float64", "int64"]:
        data[col].fillna(data[col].median(), inplace=True)
    else:
        data[col].fillna(data[col].mode()[0], inplace=True)


print("Remaining missing values:\n", data.isnull().sum())


data.reset_index(drop=True, inplace=True)



print(data.head())

data = data.dropna()
print(data.isnull().sum())


plt.figure(figsize=(8, 6))
sns.countplot(x="Potability", data=data, palette="Set2")
plt.title("Distribution of Unsafe (0) and Safe (1) Water")
plt.xlabel("Potability")
plt.ylabel("Count")
plt.show()


import plotly.express as px
figure = px.histogram(
    data,
    x="ph",
    color="Potability",
    title="Factors Affecting Water Quality: PH",
    nbins=50  
)
figure.show()

figure = px.histogram(data, x = "Hardness", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Hardness")
figure.show()

figure = px.histogram(data, x = "Solids", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Solids")
figure.show()


figure = px.histogram(data, x = "Chloramines", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Chloramines")
figure.show()

figure = px.histogram(data, x = "Sulfate", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Sulfate")
figure.show()

figure = px.histogram(data, x = "Conductivity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Conductivity")
figure.show()

figure = px.histogram(data, x = "Organic_carbon", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Organic Carbon")
figure.show()
igure = px.histogram(data, x = "Trihalomethanes", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Trihalomethanes")
figure.show()

figure = px.histogram(data, x = "Turbidity", 
                      color = "Potability", 
                      title= "Factors Affecting Water Quality: Turbidity")
figure.show()

correlation = data.corr()
correlation["ph"].sort_values(ascending=False)


from pycaret.classification import setup, compare_models


clf = setup(
    data=data,
    target="Potability",
    session_id=786,
    normalize=True,

)

best_model = compare_models()
model = create_model("rf")
predict = predict_model(model, data=data)
predict.head()

