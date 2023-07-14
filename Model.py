from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import joblib

# Đọc dữ liệu từ file Excel
data = pd.read_excel('DATA LAPTOP.xlsx')
print(data.shape)

# Tạo các LabelEncoder cho từng cột
le_Brand = LabelEncoder()
le_CPU = LabelEncoder()
le_RAM = LabelEncoder()
le_VGA = LabelEncoder()

# Mã hóa từng cột
data["Brand"] = le_Brand.fit_transform(data["Brand"])
data["CPU"] = le_CPU.fit_transform(data["CPU"])
data["RAM"] = le_RAM.fit_transform(data["RAM"])
data["VGA"] = le_VGA.fit_transform(data["VGA"])

# Tách dữ liệu thành đặc trưng (X) và biến mục tiêu (y)
X = data[['Brand', 'CPU', 'RAM', 'STORAGE', 'SCREEN', 'VGA']]
y = data['Price']

scaler = StandardScaler()
y_scaled = scaler.fit_transform(y.values.reshape(-1, 1))

X_train, X_test, y_train, y_test = train_test_split(X, y_scaled, test_size=0.2, random_state=42)

# Tạo mô hình XGBoost
xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

# Lưu các đối tượng LabelEncoder và mô hình đã huấn luyện vào các file
joblib.dump(le_Brand, 'label_encoder_brand.pkl')
joblib.dump(le_CPU, 'label_encoder_cpu.pkl')
joblib.dump(le_RAM, 'label_encoder_ram.pkl')
joblib.dump(le_VGA, 'label_encoder_vga.pkl')
joblib.dump(xgb_model, 'XGBoost.pkl')


# Hàm dự đoán giá laptop dựa trên thông tin người dùng nhập vào
def predict_price(brand, cpu, ram, storage, screen, vga):
    input_data = pd.DataFrame({'Brand': [brand], 'CPU': [cpu], 'RAM': [ram], 'STORAGE': [storage], 'SCREEN': [screen], 'VGA': [vga]})

    # Mã hóa thông tin người dùng nhập vào
    input_data["Brand"] = le_Brand.transform(input_data["Brand"])
    input_data["CPU"] = le_CPU.transform(input_data["CPU"])
    input_data["RAM"] = le_RAM.transform(input_data["RAM"])
    input_data["VGA"] = le_VGA.transform(input_data["VGA"])

    # Dự đoán giá của laptop
    y_pred = xgb_model.predict(input_data)
    y_pred_original = scaler.inverse_transform(y_pred.reshape(-1, 1))

    # Tìm laptop có giá rẻ nhất
    cheapest_product = data.loc[data['Price'].idxmin()]
    store = cheapest_product['Store']
    link = cheapest_product['LINK']

    return y_pred_original, store, link



