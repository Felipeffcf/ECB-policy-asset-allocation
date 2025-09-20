"""ECB policy & asset allocation

**ECB Monetary Policy and Bank Asset Allocation: A Correlation Analysis of Government Bonds, Corporate Lending, and Interest Rates**

Does ECB Policy Drive Banks to Prefer Corporate Loans Over Government Bonds?

Author: Felipe Fernando Calvo de Freitas

The programming format of this project was adapted for Google Collab.

"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from pandasdmx import Request
from pandas_datareader import data as pdr
from datetime import datetime
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests



# ECB SDMX request
ecb = Request('ECB')

# Define date range under analysis
start = '2010-01-01'
end = datetime.today().strftime('%Y-%m-%d')

# VARIABLES:
# Loans to Non-Financial Corportations (Corp_Loans): Loans vis-a-vis euro area households reported by MFIs excl. ESCB in the euro area (stocks), Euro area (changing composition), Monthly
# Gov Bond Holdings: Holdings of Debt securities issued by euro area General Government reported by MFIs excl. ESCB in the euro area (stocks), Euro area (changing composition), Monthly
# ECB Main Refinancing Rate: Main refinancing operations - fixed rate tenders (fixed rate) (date of changes) - Level, Euro area (changing composition), Daily

# VAR_1: Loans to Non-Financial Corporations (Millions of EUR)
loan_response = ecb.data('BSI', key='M.U2.N.A.A20.A.1.U2.2250.Z01.E', params={'startPeriod': start, 'endPeriod': end})
loan_series = loan_response.to_pandas()
# Extract TIME_PERIOD from MultiIndex and convert to DataFrame
loan_df = loan_series.reset_index()
loan_df = loan_df[['TIME_PERIOD', loan_series.name]]
loan_df.columns = ['Date', 'Corp_Loans']
loan_df['Date'] = pd.to_datetime(loan_df['Date'])
loan_df = loan_df.set_index('Date').resample('ME').mean()

# VAR_2: Gov Bond Holdings (Millions of EUR)
gov_response = ecb.data('BSI', key='M.U2.N.A.A30.A.1.U2.2100.Z01.E', params={'startPeriod': start, 'endPeriod': end})
gov_series = gov_response.to_pandas()
# Extract TIME_PERIOD from MultiIndex and convert to DataFrame
gov_df = gov_series.reset_index()
gov_df = gov_df[['TIME_PERIOD', gov_series.name]]
gov_df.columns = ['Date', 'Gov_Bonds']
gov_df['Date'] = pd.to_datetime(gov_df['Date'])
gov_df = gov_df.set_index('Date').resample('ME').mean()

# VAR_3: ECB Main Refinancing Rate (% per annum)
refi_response = ecb.data('FM', key='D.U2.EUR.4F.KR.MRR_FR.LEV', params={'startPeriod': start, 'endPeriod': end})
refi_series = refi_response.to_pandas()
# Extract TIME_PERIOD from MultiIndex and convert to DataFrame
refi_df = refi_series.reset_index()
refi_df = refi_df[['TIME_PERIOD', refi_series.name]]
refi_df.columns = ['Date', 'Refi_Rate']
refi_df['Date'] = pd.to_datetime(refi_df['Date'])
refi_df = refi_df.set_index('Date').resample('ME').mean()

# Merge all datasets
df = pd.concat([loan_df, gov_df, refi_df], axis=1).dropna()

df.to_excel('merged_data.xlsx')

# Plot time series for all variables
ax = df.plot(subplots=True, figsize=(12, 9), title='ECB Data: Corp Loans, Gov Bonds, Refi Rate')
plt.tight_layout()
plt.savefig('time_series_plot.jpg', format='jpg')
plt.show()

# Correlation matrix for our 3 variables
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.savefig('correlation_heatmap.jpg', format='jpg')
plt.show()

# Heatmap of correlations
plt.figure(figsize=(6, 4))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Scatterplot for potential crowding-out effect
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='Gov_Bonds', y='Corp_Loans')
plt.title("Crowding-Out Hypothesis")
plt.xlabel("Gov Bonds")
plt.ylabel("Corporate Loans")
plt.savefig('scatterplot_crowding_out.jpg', format='jpg')
plt.show()


# OLS regression
X = df[['Gov_Bonds', 'Refi_Rate']]
X = sm.add_constant(X)
y = df['Corp_Loans']

model = sm.OLS(y, X).fit()
print(model.summary())

# More visual version
print(model.summary().tables[1])


# Granger causality tests: Does X Granger-cause Y? Testing 6 lags
max_lags = 6
print("\n=== Granger Causality: Do Gov_Bonds & Refi_Rate 'cause' Corp_Loans? ===")
grangercausalitytests(df[['Corp_Loans', 'Gov_Bonds']], maxlag=max_lags, verbose=True)
grangercausalitytests(df[['Corp_Loans', 'Refi_Rate']], maxlag=max_lags, verbose=True)


# Stationarity check
def adf_test(series, title=''):
    result = adfuller(series.dropna(), autolag='AIC')
    print(f'ADF Test for {title}')
    print(f'  ADF Statistic: {result[0]}')
    print(f'  p-value: {result[1]}')
    print(f'  Critical Values: {result[4]}')
    print("  Stationary" if result[1] < 0.05 else "  Not stationary", "\n")

# Run ADF test for all variables
for col in df.columns:
    adf_test(df[col], col)

# First-differencing if necessary
df_diff = df.diff().dropna()

# Fit VAR model
model = VAR(df_diff)
lag_order = model.select_order(12)
print("\nSelected Lag Order (AIC):")
print(lag_order.summary())

results = model.fit(lag_order.aic)
print(results.summary())

# Impulse response functions (IRFs)
fig = irf.plot(orth=False)
fig.suptitle("Impulse Response Functions (Non-Orthogonal)")
fig.tight_layout()
fig.savefig("irf_plot.jpg", format="jpg", dpi=300)
plt.show()

