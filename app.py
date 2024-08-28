import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

# Hill equation for fitting
def dose_response(x, bottom, top, ic50, hill):
    """4-parameter logistic function for dose-response curve"""
    return bottom + (top - bottom) / (1 + (x / ic50) ** hill)


# Function to compute IC50
def compute_ic50(x, y):
    # Initial parameter guess
    p0 = [np.min(y), np.max(y), np.median(x), 1]    
    popt, _ = curve_fit(dose_response, x, y, p0=p0)
    IC50 = popt[2]
    hill = popt[3]
    return IC50, hill


st.title("IC50 Calculator")
description = r""" 

This application computes the IC50 for a substance by fitting observed effect values and substance concentrations to the Hill equation. 
The Hill equation with minimum and maximum values is given by is given by:


<div style="text-align: center;">

$$ E = E_{min} + \frac{(E_{max} - E_{min}) \cdot [Drug]^n}{[Drug]^n + IC_{50}^n} $$
</div>


Where:
- $ E $ is the effect.
- $ E_{min} $ is the minimum effect.
- $ E_{max} $ is the maximum effect.
- $ [Drug] $ is the concentration of the drug.
- $ IC_{50} $ is the concentration of the drug that produces 50% of the maximum effect.
- $ n $ is the Hill coefficient, which reflects the steepness of the curve.
"""
st.markdown(description, unsafe_allow_html=True)


# Sidebar for file upload and column selection
st.sidebar.header("Controls")
file = st.sidebar.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if file is not None:
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("Data Preview:")
    st.write(df)

    # Column selection
    x_column = st.sidebar.selectbox("Select X column", df.columns,0 )
    y_column = st.sidebar.selectbox("Select Y column", df.columns,1)

    if x_column and y_column:
        x = df[x_column].values
        y = df[y_column].values

        # Compute IC50
        ic50,hill = compute_ic50(x, y)
        
        markdown_content = f""" 
        ## Computed IC50: {f"{ic50:.3f}"} 
        """
        st.markdown(markdown_content)
        
        # Generate extrapolated data along the curve
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = dose_response(x_fit, *curve_fit(dose_response, x, y, p0=[np.min(y), np.max(y), np.median(x), 1])[0])

        # Data visualization
        fig, ax = plt.subplots()
        sns.scatterplot(x=x, y=y, label='Data points',ax=ax)
        sns.lineplot(x=x_fit, y=y_fit, label='Fitted curve', color='red', ax=ax)
        sns.despine()
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        ax.legend()
        st.pyplot(fig)
        
