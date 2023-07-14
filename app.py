from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

app = Flask(__name__)

# Load trained models and label encoders
le_Brand = joblib.load('label_encoder_brand.pkl')
le_CPU = joblib.load('label_encoder_cpu.pkl')
le_VGA = joblib.load('label_encoder_vga.pkl')

xgb_model = joblib.load('XGBoost.pkl')

# Load data
data = pd.read_excel('DATA LAPTOP.xlsx')

# Preprocess data
scaler = StandardScaler()

X = data[['Brand', 'CPU', 'RAM', 'STORAGE', 'SCREEN', 'VGA']]
y = data['Price']
X_encoded = X.copy()
y = y.values.reshape(-1, 1)
scaler.fit(y)
y = pd.DataFrame(scaler.transform(y))

# Encode categorical features
X_encoded['Brand'] = le_Brand.transform(X['Brand'])
X_encoded['CPU'] = le_CPU.transform(X['CPU'])
X_encoded['VGA'] = le_VGA.transform(X['VGA'])

# # Perform one-hot encoding for encoded features
# X_encoded = pd.get_dummies(X_encoded, columns=['Brand', 'CPU', 'VGA'])

# Scale features
# X_scaled = scaler.fit_transform(X_encoded)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form')
def form():
    return render_template('form.html')

@app.route('/result', methods=['POST'])
def result():
    Brand = request.form.get('Brand')
    CPU = request.form.get('CPU')
    RAM = int(request.form.get('RAM'))
    STORAGE = int(request.form.get('STORAGE'))
    VGA = request.form.get('VGA')
    SCREEN = float(request.form.get('SCREEN'))

    # Encode input features
    try:
        brand_encoded = le_Brand.transform([Brand])[0]
        cpu_encoded = le_CPU.transform([CPU])[0]
        vga_encoded = le_VGA.transform([VGA])[0]
    except ValueError as e:
        return render_template('error.html', error_message=str(e))

    input_encoded = pd.DataFrame({'Brand': [brand_encoded],'CPU': [cpu_encoded], 'RAM': [RAM], 'STORAGE': [STORAGE], 'SCREEN': [SCREEN],'VGA': [vga_encoded]})
    # input_encoded = pd.get_dummies(input_data, columns=['Brand', 'CPU', 'VGA'])

    # Keep only the necessary columns for prediction
    input_encoded = input_encoded.reindex(columns=X_encoded.columns, fill_value=0)



    # Make prediction
    predicted_price = xgb_model.predict(input_encoded)
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))[0][0]
    closest_row_index = abs(data['Price'] - predicted_price).idxmin()
    closest_store = data.loc[closest_row_index, 'Store']
    closest_link = data.loc[closest_row_index, 'LINK']

# Truyền thông tin cửa hàng và liên kết vào template
    return render_template('result.html', predicted_price=round(predicted_price, 2), store=closest_store, link_store=closest_link)

if __name__ == '__main__':
    app.run(debug=True)


