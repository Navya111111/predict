from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessor
with open('sales_model.pkl', 'rb') as f:
    model, transformer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        input_data = {
            'Product_Category': request.form['product_category'],
            'Store_Location': request.form['store_location'],
            'Day_of_Week': request.form['day_of_week'],
            'Holiday_Week': int(request.form['holiday_week']),
            'Promotion_Applied': int(request.form['promotion_applied']),
            'Price': float(request.form['price']),
            'Last_Week_Sales': float(request.form['last_week_sales'])
        }

        # Convert to DataFrame
        df_input = pd.DataFrame([input_data])

        # Transform
        transformed_input = transformer.transform(df_input)

        # Predict
        prediction = model.predict(transformed_input)

        return render_template('index.html', prediction_text=f"Predicted Units Sold: {round(prediction[0], 2)}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)