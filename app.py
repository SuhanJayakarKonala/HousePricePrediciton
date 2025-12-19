import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("🏠 Housing Price Prediction App")
st.markdown("---")

# Load or train the model
@st.cache_resource
def load_model():
    model_path = 'linear_regression_model.pkl'
    
    # Check if model exists
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model, None
    else:
        # Train a new model if not exists
        df = pd.read_csv('Housing.csv')
        
        # One-hot encode categorical columns
        categorical_cols = df.select_dtypes(include='object').columns
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Prepare features and target
        X = df_encoded.drop('price', axis=1)
        y = df_encoded['price']
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Save model
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        
        return model, X.columns.tolist()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Housing.csv')
    return df

# Load model and data
model, feature_columns = load_model()
df = load_data()

# Get feature names from the dataset
if feature_columns is None:
    # If model was loaded from pickle, we need to recreate the feature columns
    categorical_cols = df.select_dtypes(include='object').columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    feature_columns = [col for col in df_encoded.columns if col != 'price']

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Price Prediction", "Dataset Overview"])

if page == "Price Prediction":
    st.header("Predict House Price")
    st.write("Enter the house details below to get a price prediction:")
    
    # Create two columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📏 Physical Features")
        area = st.number_input("Area (sq ft)", min_value=1000, max_value=20000, value=5000, step=100)
        bedrooms = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=1)
        stories = st.selectbox("Stories", options=[1, 2, 3, 4], index=1)
        parking = st.selectbox("Parking Spaces", options=[0, 1, 2, 3], index=1)
    
    with col2:
        st.subheader("🏡 Amenities & Features")
        mainroad = st.selectbox("Main Road Access", options=["yes", "no"], index=0)
        guestroom = st.selectbox("Guest Room", options=["yes", "no"], index=1)
        basement = st.selectbox("Basement", options=["yes", "no"], index=1)
        hotwaterheating = st.selectbox("Hot Water Heating", options=["yes", "no"], index=1)
        airconditioning = st.selectbox("Air Conditioning", options=["yes", "no"], index=0)
        prefarea = st.selectbox("Preferred Area", options=["yes", "no"], index=0)
        furnishingstatus = st.selectbox("Furnishing Status", 
                                       options=["furnished", "semi-furnished", "unfurnished"], 
                                       index=0)
    
    # Predict button
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        # Create input dataframe
        input_data = {
            'area': area,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'stories': stories,
            'parking': parking,
            'mainroad_yes': 1 if mainroad == 'yes' else 0,
            'guestroom_yes': 1 if guestroom == 'yes' else 0,
            'basement_yes': 1 if basement == 'yes' else 0,
            'hotwaterheating_yes': 1 if hotwaterheating == 'yes' else 0,
            'airconditioning_yes': 1 if airconditioning == 'yes' else 0,
            'prefarea_yes': 1 if prefarea == 'yes' else 0,
            'furnishingstatus_semi-furnished': 1 if furnishingstatus == 'semi-furnished' else 0,
            'furnishingstatus_unfurnished': 1 if furnishingstatus == 'unfurnished' else 0
        }
        
        # Create dataframe with all features in correct order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match model's expected order
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Display result
        st.markdown("---")
        st.success("✅ Prediction Complete!")
        
        # Display prediction in a nice format
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.metric(
                label="Predicted House Price",
                value=f"₹{prediction:,.2f}",
                delta=None
            )
            
            # Additional info
            st.info(f"💰 Approximately: ₹{prediction/100000:.2f} Lakhs")
        
        # Show input summary
        with st.expander("📋 Input Summary"):
            summary_col1, summary_col2 = st.columns(2)
            with summary_col1:
                st.write("**Physical Features:**")
                st.write(f"- Area: {area} sq ft")
                st.write(f"- Bedrooms: {bedrooms}")
                st.write(f"- Bathrooms: {bathrooms}")
                st.write(f"- Stories: {stories}")
                st.write(f"- Parking: {parking}")
            
            with summary_col2:
                st.write("**Amenities:**")
                st.write(f"- Main Road: {mainroad}")
                st.write(f"- Guest Room: {guestroom}")
                st.write(f"- Basement: {basement}")
                st.write(f"- Hot Water: {hotwaterheating}")
                st.write(f"- AC: {airconditioning}")
                st.write(f"- Preferred Area: {prefarea}")
                st.write(f"- Furnishing: {furnishingstatus}")

elif page == "Dataset Overview":
    st.header("📊 Dataset Overview")
    
    # Display basic stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Houses", len(df))
    with col2:
        st.metric("Average Price", f"₹{df['price'].mean()/100000:.2f}L")
    with col3:
        st.metric("Min Price", f"₹{df['price'].min()/100000:.2f}L")
    with col4:
        st.metric("Max Price", f"₹{df['price'].max()/100000:.2f}L")
    
    st.markdown("---")
    
    # Show sample data
    st.subheader("Sample Data")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Show statistics
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
    st.markdown("---")
    
    # Visualizations
    st.subheader("📈 Data Visualizations")
    
    # Price Distribution
    st.markdown("#### Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df['price']/100000, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Price (in Lakhs)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of House Prices', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Numerical features analysis
    st.markdown("#### Relationship between Features and Price")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Area vs Price
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['area'], df['price']/100000, alpha=0.6, color='coral')
        ax.set_xlabel('Area (sq ft)', fontsize=11)
        ax.set_ylabel('Price (in Lakhs)', fontsize=11)
        ax.set_title('Area vs Price', fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Bedrooms vs Price
        fig, ax = plt.subplots(figsize=(8, 6))
        bedroom_price = df.groupby('bedrooms')['price'].mean()/100000
        ax.bar(bedroom_price.index, bedroom_price.values, color='lightgreen', edgecolor='black')
        ax.set_xlabel('Number of Bedrooms', fontsize=11)
        ax.set_ylabel('Average Price (in Lakhs)', fontsize=11)
        ax.set_title('Average Price by Bedrooms', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Categorical features impact
    st.markdown("#### Impact of Amenities on Price")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # AC impact
        fig, ax = plt.subplots(figsize=(8, 6))
        ac_price = df.groupby('airconditioning')['price'].mean()/100000
        colors = ['#ff9999', '#66b3ff']
        ax.bar(ac_price.index, ac_price.values, color=colors, edgecolor='black')
        ax.set_xlabel('Air Conditioning', fontsize=11)
        ax.set_ylabel('Average Price (in Lakhs)', fontsize=11)
        ax.set_title('Impact of AC on Price', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        # Furnishing status impact
        fig, ax = plt.subplots(figsize=(8, 6))
        furnish_price = df.groupby('furnishingstatus')['price'].mean()/100000
        colors = ['#99ff99', '#ffcc99', '#ff99cc']
        ax.bar(furnish_price.index, furnish_price.values, color=colors, edgecolor='black')
        ax.set_xlabel('Furnishing Status', fontsize=11)
        ax.set_ylabel('Average Price (in Lakhs)', fontsize=11)
        ax.set_title('Impact of Furnishing on Price', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=15)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.markdown("#### Feature Correlations")
    
    # Prepare data for correlation
    df_corr = df.copy()
    # Convert yes/no to 1/0
    binary_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
    for col in binary_cols:
        df_corr[col] = (df_corr[col] == 'yes').astype(int)
    
    # One-hot encode furnishing status
    df_corr = pd.get_dummies(df_corr, columns=['furnishingstatus'], drop_first=True)
    
    # Calculate correlation matrix
    corr_matrix = df_corr.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature distribution for categorical variables
    st.markdown("#### Categorical Feature Distributions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        mainroad_counts = df['mainroad'].value_counts()
        colors = ['#90EE90', '#FFB6C1']
        ax.pie(mainroad_counts.values, labels=mainroad_counts.index, autopct='%1.1f%%',
               startangle=90, colors=colors)
        ax.set_title('Main Road Access Distribution', fontsize=12, fontweight='bold')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        prefarea_counts = df['prefarea'].value_counts()
        colors = ['#87CEEB', '#FFD700']
        ax.pie(prefarea_counts.values, labels=prefarea_counts.index, autopct='%1.1f%%',
               startangle=90, colors=colors)
        ax.set_title('Preferred Area Distribution', fontsize=12, fontweight='bold')
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Information
    st.subheader("Feature Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Features:**")
        st.write("- price: House price")
        st.write("- area: Plot area in square feet")
        st.write("- bedrooms: Number of bedrooms")
        st.write("- bathrooms: Number of bathrooms")
        st.write("- stories: Number of stories")
        st.write("- parking: Number of parking spaces")
    
    with col2:
        st.write("**Categorical Features:**")
        st.write("- mainroad: Connected to main road (yes/no)")
        st.write("- guestroom: Has guest room (yes/no)")
        st.write("- basement: Has basement (yes/no)")
        st.write("- hotwaterheating: Has hot water heating (yes/no)")
        st.write("- airconditioning: Has AC (yes/no)")
        st.write("- prefarea: In preferred area (yes/no)")
        st.write("- furnishingstatus: furnished/semi-furnished/unfurnished")