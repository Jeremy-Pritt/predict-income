import streamlit as st
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

st.header('Predicting Income')

education_df = pd.read_csv(r"education.csv")
unemployment_df = pd.read_csv(r"unemployment.csv")
df = pd.merge(left=education_df, right=unemployment_df,
              left_on="FIPS Code", right_on="FIPS_Code")

# remove commas and change the median household income to datatype float
df["Median_Household_Income_2019"] = df["Median_Household_Income_2019"].str.replace(
    ",", "")
df["Median_Household_Income_2019"] = df["Median_Household_Income_2019"].astype(
    float)

# population type for subsetting data
# pop_type = df["City/Suburb/Town/Rural 2013"]

# replace long column names for variables of interest
df.rename(columns={
    'Percent of adults with less than a high school diploma, 2015-19': 'no_hs_diploma'}, inplace=True)
df.rename(columns={
    'Percent of adults with a high school diploma only, 2015-19': 'hs_diploma'}, inplace=True)
df.rename(columns={
    'Percent of adults completing some college or associate\'s degree, 2015-19': 'some_college'}, inplace=True)
df.rename(columns={
    'Percent of adults with a bachelor\'s degree or higher, 2015-19': 'bachelors_plus'}, inplace=True)
df.rename(
    columns={'City/Suburb/Town/Rural 2013': 'pop_type'}, inplace=True)
df.rename(
    columns={'Median_Household_Income_2019': 'median_income'}, inplace=True)
df.rename(
    columns={'Unemployment_rate_2020': 'unemployment_rate'}, inplace=True)

# create a final df with just the data we need and change pop_type values to either urban or rural
df["pop_type"] = df["pop_type"].replace(['City'], 'Urban')
df["pop_type"] = df["pop_type"].replace(['Town'], 'Urban')
df["pop_type"] = df["pop_type"].replace(['Suburb'], 'Urban')
df_final = df[["no_hs_diploma", "hs_diploma", "some_college",
               "bachelors_plus", "pop_type", "median_income", "unemployment_rate"]]

# run the multiple linear regresion
results = smf.ols(
    'median_income ~ pop_type + no_hs_diploma + hs_diploma + some_college', data=df_final).fit()

# display the residuals to consider regression assumptions
fig, ax = plt.subplots(figsize=(15, 10))
resid_fig = sm.graphics.plot_regress_exog(results, 'hs_diploma', fig=fig)

# There appears to be some non-linearity in the data. I will try and fix this by using the log(response variable)
results2 = smf.ols(
    'np.log(median_income) ~ pop_type + no_hs_diploma + hs_diploma + some_college', data=df_final).fit()

# display the new model of the residuals
fig, ax = plt.subplots(figsize=(15, 10))
resid_fig2 = sm.graphics.plot_regress_exog(results2, 'hs_diploma', fig=fig)

# streamlit features
if st.button('Step 1: Import Data'):
    st.write('Education data and employment data uploaded.')

    st.write('Total Observations:', len(df.index))
    st.write('Total Variables:', len(df.columns))


if st.button('Step 2: Clean the Data'):
    st.write("Superfluous variables removed. Datatypes cleaned and commas removed.")
    st.write('Total Observations:', len(df_final.index))
    st.write('Total Variables:', len(df_final.columns))

if st.button('Step 3: View Dataframe Head'):
    st.write(df_final.head())

if st.button('Step 4: Run Multiple Regression'):
    st.write("Response Variable: Income")
    st.write("Explanatory Variables: Level of Education, Rural vs Urban")
    st.write(results.summary())


if st.button('Step 5: Check the Residuals'):
    st.pyplot(fig=resid_fig, clear_figure=True)

if st.button('Step 6: Perform Transformations to Fix Nonlinearity'):
    st.write(
        'New residuals plotted after using log transformation on response variable:')
    st.pyplot(fig=resid_fig2, clear_figure=True)

if st.button('Step 7: Final Multiple Regresion Results'):
    st.write(results2.summary())

if st.button('Step 8: View Interpretation'):
    st.write("With a large F-statistic of 879.8 and significant p-values, education level and/or population type (urban or rural) appear to have an impact on expected median income.")
    st.write("Having a no HS Diploma will result in a -0.0212 unit decrease in the log(median income) while holding population type (urban or rural) fixed. This means that someone with no HS Diploma, in an area where the median income is 60,000, would be expected to have a lower median income by about $15,569.23.")
    st.write("Having a HS Diploma with no higher level of education will result in a - 0.0137 unit decrease in the log(median income) while holding population type(urban or rural) fixed. This means that someone with only a HS Diploma, in an area where the median income is 60,000 would be expected to have a lower median income by about 13, 984.84.")
    st.write("Having some college with no higher level of education will result in a - 0.0101 unit decrease in the log(median income) while holding population type(urban or rural) fixed. This means that someone with only some college, in an area where the median income is 60,000, would be expected to have a lower median income by about 13, 057.05.")
    st.write("Living in an urban area will result in a 0.0665 unit increase in the log(median income) while holding level of education fixed. This means that someone living in an urban area, where the median income is 60,000, would be expected to have a larger higher median income of about 22,135.70 compared to if it were a rural area.")
