import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def app():
    """Visualize house price distributions and feature relationships."""
    st.title("Price and Feature Visualization")

    data_path = "data/final_cleaned_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please run the data cleaning process first.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Price distribution (histogram and box plot)
    st.write("### SalePrice Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(data["SalePrice"], bins=30, ax=ax)
        ax.set_title("Histogram of Sale Prices")
        ax.set_xlabel("Sale Price ($)")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data["SalePrice"], ax=ax)
        ax.set_title("Box Plot of Sale Prices")
        ax.set_ylabel("Sale Price ($)")
        st.pyplot(fig)

    # Feature vs. SalePrice scatter plots (interactive selection)
    st.write("### Feature vs. SalePrice Scatter Plots")
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    selected_feature = st.selectbox("Select a Feature", options=numeric_features, index=numeric_features.get_loc("GrLivArea"))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x=selected_feature, y="SalePrice", ax=ax)
    ax.set_title(f"{selected_feature} vs. Sale Price")
    ax.set_xlabel(selected_feature)
    ax.set_ylabel("Sale Price ($)")
    st.pyplot(fig)

    # Top 10 correlated features with SalePrice (bar plot)
    st.write("### Top 10 Features Correlated with SalePrice")
    correlation_with_price = data[numeric_features].corr()["SalePrice"].sort_values(ascending=False)[1:11]  # Exclude SalePrice itself
    fig, ax = plt.subplots(figsize=(10, 6))
    correlation_with_price.plot(kind="bar", ax=ax)
    ax.set_title("Top 10 Features Correlated with Sale Price")
    ax.set_xlabel("Feature")
    ax.set_ylabel("Correlation Coefficient")
    st.pyplot(fig)

if __name__ == "__main__":
    app()