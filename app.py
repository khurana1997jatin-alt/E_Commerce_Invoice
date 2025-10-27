import streamlit as st
import pandas as pd
import pickle as pkl

# 🚀 Load model
MODEL_PATH = 'E_Commerce_Invoice_model9.pkl'
model = pkl.load(open(MODEL_PATH, 'rb'))

# ✅ Define expected country columns from training
expected_country_cols = [
    'Country_Australia', 'Country_Austria', 'Country_Bahrain', 'Country_Belgium',
    'Country_Brazil', 'Country_Canada', 'Country_Channel Islands', 'Country_Cyprus',
    'Country_Czech Republic', 'Country_Denmark', 'Country_EIRE', 'Country_European Community',
    'Country_Finland', 'Country_France', 'Country_Germany', 'Country_Greece',
    'Country_Hong Kong', 'Country_Iceland', 'Country_Israel', 'Country_Italy',
    'Country_Japan', 'Country_Lebanon', 'Country_Lithuania', 'Country_Malta',
    'Country_Netherlands', 'Country_Norway', 'Country_Poland', 'Country_Portugal',
    'Country_RSA', 'Country_Saudi Arabia', 'Country_Singapore', 'Country_Spain',
    'Country_Sweden', 'Country_Switzerland', 'Country_USA', 'Country_United Arab Emirates',
    'Country_United Kingdom', 'Country_Unspecified'
]

# 🖼️ Streamlit UI setup
st.set_page_config(page_title="Retail Invoice Predictor", layout="centered")
st.title('🛒 Retail Invoice Price Prediction')
st.write("Fill in the invoice details below to estimate the total invoice price.")

# 📅 Temporal Inputs
st.subheader("📅 Invoice Timing")
month = st.number_input('Invoice Month', min_value=1, max_value=12)
year = st.number_input('Invoice Year', min_value=2009, max_value=2025)
day = st.number_input('Day of Month', min_value=1, max_value=31)
hour = st.number_input('Hour of Day', min_value=0, max_value=23)
weekday = st.number_input('Weekday (0=Mon, 6=Sun)', min_value=0, max_value=6)

# 🌍 Country One-Hot Inputs
st.subheader("🌍 Country")
country_labels = {col: col.replace('Country_', '').replace('_', ' ') for col in expected_country_cols}
label_to_column = {v: k for k, v in country_labels.items()}

selected_label = st.selectbox('Select Country', list(label_to_column.keys()))
selected_country = label_to_column[selected_label]
country_vector = [1 if col == selected_country else 0 for col in expected_country_cols]

# 📦 Invoice Details
st.subheader("📦 Invoice Details")
quantity = st.number_input('Quantity', min_value=0.0)
unit_price = st.number_input('Unit Price', min_value=0.0)
total_price = st.number_input('Total Price', min_value=0.0)
total_product_qty = st.number_input('Total Product Quantity', min_value=0.0)
total_invoice_qty = st.number_input('Total Invoice Quantity', min_value=0.0)

# 🧮 Combine all inputs
input_values = [month, year, day, hour, weekday] + country_vector + [
    quantity, unit_price, total_price, total_product_qty, total_invoice_qty
]

input_columns = ['InvoiceMonth', 'InvoiceYear', 'Day', 'Hour', 'Weekday'] + expected_country_cols + [
    'Quantity', 'UnitPrice', 'Total Price', 'TotalProductQuantity', 'TotalInvoiceQuantity'
]

assert len(input_columns) == len(input_values), f"Mismatch: {len(input_columns)} columns vs {len(input_values)} values"
input_df = pd.DataFrame([input_values], columns=input_columns)

# 🔮 Predict
if st.button('Predict'):
    if any(val == 0 for val in [quantity, unit_price, total_product_qty, total_invoice_qty]):
        st.warning("Please enter valid non-zero values for quantity, price, and totals.")
    else:
        try:
            prediction = model.predict(input_df)[0]
            st.success(f'💰 Predicted Total Invoice Price: ₹{prediction:,.2f}')
        except Exception as e:
            st.error(f"Prediction failed: {e}")

