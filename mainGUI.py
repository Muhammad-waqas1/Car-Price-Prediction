import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Load dataset and models
df = pd.read_csv('CAR_DETAILS.csv')
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

numerical_features = ['year', 'km_driven']
categorical_features = ['fuel', 'seller_type', 'transmission', 'owner']

numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

models = {
    'Random Forest Regression': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Initialize Tkinter
root = tk.Tk()
root.title("Car Price Prediction")
root.geometry("700x700")
root.configure(bg='#f5faf5')

# Styling
style = ttk.Style()
style.configure('TLabel', background='#c41a64', font=('Arial', 12))
style.configure('TButton', background='#c41a64', foreground='#040405', font=('Arial', 14, 'bold'))
style.map('TButton', background=[('active', '#004d40')])
style.configure('TEntry', fieldbackground='#ffffff', font=('Arial', 12))
style.configure('TCombobox', font=('Arial', 12))

# Function to validate year
def validate_year(year):
    return year.isdigit() and len(year) == 4

# Function to clear all input fields
def reset_fields():
    for field in vars:
        if isinstance(vars[field], ttk.Combobox):
            vars[field].set('')  # Clear Combobox fields
        elif isinstance(vars[field], tk.Entry):
            vars[field].delete(0, tk.END)  # Clear Entry fields
    result_label.config(text="")  # Clear prediction result

# Function to predict price
def predict_price():
    # Clear the previous result
    result_label.config(text="")
    
    data = {
        'name': name_var.get(),
        'year': int(year_var.get()),
        'km_driven': int(km_var.get()),
        'fuel': fuel_var.get(),
        'seller_type': seller_var.get(),
        'transmission': trans_var.get(),
        'owner': owner_var.get()
    }
    input_df = pd.DataFrame([data])
    X_input = input_df.drop(['name'], axis=1)
    
    # Choose model (for demonstration, we'll use Gradient Boosting)
    model = models['Gradient Boosting Regression']
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    
    # Train the model
    pipeline.fit(X, y)
    
    # Predict
    prediction = pipeline.predict(X_input)
    result_label.config(text=f"Predicted Price: ${prediction[0]:,.2f}", fg='green')

# UI Components
title_label = tk.Label(root, text="Car Price Prediction", font=("Arial", 24, "bold"), bg='#9ce0ff', foreground='#040405')
title_label.grid(row=0, column=0, columnspan=2, pady=20)

fields = ['Name', 'Year', 'KM Driven', 'Fuel', 'Seller Type', 'Transmission', 'Owner']
vars = {}

from CarNames import Names

row = 1
col = 0

for i, field in enumerate(fields):
    if i % 2 == 0 and i != 0:
        row += 1
        col = 0
    
    label = tk.Label(root, text=field, bg='#f5faf5')
    label.grid(row=row, column=col, padx=10, pady=5, sticky='w')
    
    if field == 'Name':
        vars[field] = ttk.Combobox(root, values=Names)
    elif field == 'Year':
        vars[field] = tk.Entry(root, validate='key', validatecommand=(root.register(lambda P: P.isdigit() or P == ""), '%P'))
    elif field == 'KM Driven':
        vars[field] = tk.Entry(root, validate='key', validatecommand=(root.register(lambda P: P.isdigit() or P == ""), '%P'))
    else:
        vars[field] = ttk.Combobox(root, values=['Petrol', 'Diesel', 'CNG'] if field == 'Fuel' else
                                             ['Individual', 'Dealer'] if field == 'Seller Type' else
                                             ['Manual', 'Automatic'] if field == 'Transmission' else
                                             ['First Owner', 'Second Owner', 'Third Owner'])
    
    vars[field].grid(row=row, column=col+1, padx=10, pady=5, sticky='w', ipadx=10, ipady=5, columnspan=1)
    col += 2

name_var = vars['Name']
year_var = vars['Year']
km_var = vars['KM Driven']
fuel_var = vars['Fuel']
seller_var = vars['Seller Type']
trans_var = vars['Transmission']
owner_var = vars['Owner']

# Predict button
predict_button = ttk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=row+1, column=0, columnspan=2, pady=20)

# Reset button to clear all inputs
reset_button = ttk.Button(root, text="Reset", command=reset_fields)
reset_button.grid(row=row+2, column=0, columnspan=2, pady=10)

result_label = tk.Label(root, text="", font=("Arial", 18, "bold"), bg='#f5faf5')
result_label.grid(row=row+3, column=0, columnspan=2, pady=10)

root.mainloop()
