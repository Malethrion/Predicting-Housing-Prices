import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def app():
    """Analyze correlations and visualize house price trends."""
    st.title("Correlation Study")

    data_path = "data/final_cleaned_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please run the data cleaning process first.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_data.corr()

    # Visualize correlation heatmap
    st.write("### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

    # Price distribution
    st.write("### SalePrice Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data["SalePrice"], bins=30, ax=ax)
    ax.set_title("Distribution of Sale Prices")
    ax.set_xlabel("Sale Price ($)")
    st.pyplot(fig)

    # Scatter plot: GrLivArea vs. SalePrice
    st.write("### GrLivArea vs. SalePrice")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x="GrLivArea", y="SalePrice", ax=ax)
    ax.set_title("Living Area vs. Sale Price")
    ax.set_xlabel("Above Ground Living Area (SqFt)")
    ax.set_ylabel("Sale Price ($)")
    st.pyplot(fig)

    # Scatter plot: OverallQual vs. SalePrice
    st.write("### Overall Quality vs. SalePrice")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=data, x="OverallQual", y="SalePrice", ax=ax)
    ax.set_title("Overall Quality vs. Sale Price")
    ax.set_xlabel("Overall Quality (1-10)")
    ax.set_ylabel("Sale Price ($)")
    st.pyplot(fig)

if __name__ == "__main__":
    app()


