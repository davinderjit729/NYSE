import streamlit as st
import pandas as pd
import csv
import plotly.express as px
from PIL import Image

#opening the image
image = Image.open('C:/Users/user/Downloads/market.jpg')

# Create columns for layout
col1, col2 = st.columns([1, 1])
col1.subheader("New York Stock Exchange (NYSE), one of the world's largest marketplaces for securities and other exchange-traded investments.")
col2.image(image, caption='Stock Market')


st.header("Datasets view")
# Create columns for layout
col1, col2, col3 = st.columns([1, 1,1])


# Load data
prices_path = 'C:/Users/user/Downloads/prices.csv'
prices = pd.read_csv(prices_path, encoding='utf-8')

fundamentals_path = 'C:/Users/user/Downloads/fundamentals.csv'
fundamentals = pd.read_csv(fundamentals_path, encoding='utf-8')

securities_path = 'C:/Users/user/Downloads/securities.csv'
securities = pd.read_csv(securities_path, encoding='utf-8')


# Display data in columns
with col1:
    show_prices = st.button("Prices")
    if show_prices:
        st.dataframe(prices)


with col2:
    show_fundamentals = st.button("Fundamentals")
    if show_fundamentals:
        st.dataframe(fundamentals)

with col3:
    show_securities = st.button("Securities")
    if show_securities:
        st.dataframe(securities)


st.write("Descriptive statistics of Prices dataset")
st.write(prices.describe())

st.write("Descriptive statistic of fundamental datatset")
st.write(fundamentals.describe())

col1, col2, col3 = st.columns([1, 2, 2])
col1.write("Null values for Prices ")
col1.dataframe(prices.isnull().sum().reset_index())
col2.write("Null values for fundamentals")
col2.dataframe(fundamentals.isnull().sum().reset_index())
col3.write("Null values for securities")
col3.dataframe(securities.isnull().sum().reset_index())


#Analysis of securities dataset 
#part1 :securities by sector on pie chart

def generate_pie_chart(securities):
    fig = px.pie(securities, names='GICS Sector', title='Distribution of Companies by Sector')
    return fig

st.plotly_chart(generate_pie_chart(securities))


# part2 :count of securities by state,
# Function to separate city and state
def separate_city_state(address):
    parts = address.split(', ')
    city = parts[0] if len(parts) > 0 else None
    state = parts[1] if len(parts) > 1 else None
    return city, state

# Create new columns 'City' and 'State'
securities[['City', 'State']] = securities['Address of Headquarters'].apply(lambda x: pd.Series(separate_city_state(x)))

#Draw map of number of securities by state .
def generate_map(securities):
    state_counts = securities['State'].value_counts().reset_index()
    state_counts.columns = ['State', 'Number of Securities']

    fig = px.choropleth_mapbox(
        state_counts,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",  
        locations="State",
        featureidkey="properties.NAME",
        color="Number of Securities",
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=3,
        center={"lat": 37.0902, "lon": -95.7129},  
        opacity=0.5,
        labels={"Number of Securities": "Number of Securities"},
        title='Number of Securities by State',
    )

    fig.update_layout(mapbox_style="carto-positron")
    return fig

st.plotly_chart(generate_map(securities))


# Analysis of price dataset for technical anlaysis
#part1
#ist : Group by company name and calculate the average of close prices
average_close_prices = prices.groupby('symbol')['close'].mean().reset_index()

# 2nd:Find the index of the latest date for each symbol
# Convert the 'date' column to datetime
prices['date'] = pd.to_datetime(prices['date'])
latest_date_index = prices.groupby('symbol')['date'].idxmax()

# Extract the rows with the latest date for each symbol
latest_prices = prices.loc[latest_date_index]

# Merge the two DataFrames on 'symbol'
merged_df = pd.merge(latest_prices, average_close_prices, on='symbol', suffixes=('_latest', '_average'))

# Apply the condition and create a new column 'recommendation'
st.title("Long term recommendation")
merged_df['recommendation'] = merged_df['close_latest'] > merged_df['close_average']
merged_df['recommendation'] = merged_df['recommendation'].apply(lambda x: 'Long-term buy' if x else 'not buy ')

# Create a button in the sidebar
Long_term_button = st.sidebar.button("Long Term Recommendation")

if Long_term_button:
    st.title("Long term recommendation")
st.dataframe(merged_df[['symbol', 'close_latest', 'close_average', 'recommendation']])
     

#part2
# Calculate the latest 50 days average of close prices for each symbol
rolling_means = []

for symbol, group in prices.groupby('symbol'):  
 rolling_mean = group['close'].rolling(window=50, min_periods=1).mean().reset_index(drop=True) 
 rolling_means.append(pd.concat([group['symbol'].reset_index(drop=True), rolling_mean], axis=1,))

# Concatenate the results
latest_ma50_by_symbol = pd.concat(rolling_means, ignore_index=True)

# Calculate the average of the latest 50 days average close prices for each symbol
average_ma50_by_symbol = latest_ma50_by_symbol.groupby('symbol')['close'].mean().reset_index()

# Merge the latest close price with the average of the latest 50 days close prices
merged_df = pd.merge(latest_prices, average_ma50_by_symbol, on='symbol')

# Determine the recommendation based on the comparison
merged_df['Recommendation'] = merged_df.apply(lambda row: 'Mid-term buy' if row['close_x'] > row['close_y'] else 'not buy', axis=1)

# Create a button in the sidebar
Mid_term_button = st.sidebar.button("Mid Term Recommendation")

if Mid_term_button:
    # Display the DataFrame in the main area
    st.title("Mid term recommendation")
    st.dataframe(merged_df[['symbol', 'close_x', 'close_y', 'Recommendation']]) 



#Part3

# Calculate the latest 20 days average of close prices for each symbol
rolling_means = []

for symbol, group in prices.groupby('symbol'):  
 rolling_mean = group['close'].rolling(window=20, min_periods=1).mean().reset_index(drop=True) 
 rolling_means.append(pd.concat([group['symbol'].reset_index(drop=True), rolling_mean], axis=1,))

# Concatenate the results
latest_ma20_by_symbol = pd.concat(rolling_means, ignore_index=True)

# Calculate the average of the latest 20 days average close prices for each symbol
average_ma20_by_symbol = latest_ma20_by_symbol.groupby('symbol')['close'].mean().reset_index()

# Merge the latest close price with the average of the latest 20 days close prices
merged_df = pd.merge(latest_prices, average_ma20_by_symbol, on='symbol')

# Determine the recommendation based on the comparison
merged_df['Recommendation'] = merged_df.apply(lambda row: 'Short-term buy' if row['close_x'] > row['close_y'] else 'not buy', axis=1)


# Create a button in the sidebar
short_term_button = st.sidebar.button("Short Term Recommendation")

if short_term_button:
    # Display the DataFrame in the main area
    st.title("Short term recommendation")
    st.dataframe(merged_df[['symbol', 'close_x', 'close_y', 'Recommendation']])      

#Part 4 Charts creation


st.title("Select Stock Name and MA20/MA50/MA200")

# Create a stock selection dropdown
selected_stock = st.selectbox("Select a Stock", prices['symbol'].unique())

# Filter data for the selected stock
selected_stock_data = prices[prices['symbol'] == selected_stock]

import plotly.graph_objects as go


# Create a moving average selection dropdown
selected_ma = st.selectbox("Select Moving Average", [20, 50, 200])

# Calculate moving average
ma_column = f'MA{selected_ma}'
selected_stock_data[ma_column] = selected_stock_data['close'].rolling(window=selected_ma).mean()

# Candlestick chart with selected moving average
fig = go.Figure()

# Candlestick trace
fig.add_trace(go.Candlestick(x=selected_stock_data['date'],
                open=selected_stock_data['open'],
                high=selected_stock_data['high'],
                low=selected_stock_data['low'],
                close=selected_stock_data['close'],
                name='Candlestick'))

# Moving average trace
fig.add_trace(go.Scatter(x=selected_stock_data['date'], y=selected_stock_data[ma_column],
                         mode='lines', line=dict(color='orange'), name=f'MA{selected_ma}'))

# Set layout    
fig.update_layout(title=f'Candlestick Chart with Moving Average ({selected_ma} days) for {selected_stock}',
                  xaxis_title='Date',
                  yaxis_title='Stock Price',
                  xaxis_rangeslider_visible=False)

# Show the chart in Streamlit
st.title(f"Candlestick Chart with Moving Average ({selected_ma} days) for {selected_stock}")
st.plotly_chart(fig)

# Fundamental analysis
#Replace null values for columns Cash Ratio,quick ratio,Current Ratio,Earnings Per Share,Estimated Shares Outstanding,For Year
fundamentals = fundamentals.loc[:, ~fundamentals.columns.str.contains('^Unnamed')]
fundamentals['For Year'] = fundamentals['Period Ending']



def calculate_cash_ratio(row):
    if pd.isnull(row['Cash Ratio']):
        return row['Cash and Cash Equivalents'] / row['Total Liabilities']*100
    else:
        return row['Cash Ratio']

# Apply the formula to fill null values in the 'Cash Ratio' column
fundamentals['Cash Ratio'] = fundamentals.apply(calculate_cash_ratio, axis=1)


# Define the formula to calculate Current Ratio
def calculate_current_ratio(row):
    if pd.isnull(row['Current Ratio']):
        return (row['Total Current Assets'] - row['Total Current Liabilities'])*100
    else:
        return row['Current Ratio']

# Apply the formula to fill null values in the 'Current Ratio' column
fundamentals['Current Ratio'] = fundamentals.apply(calculate_current_ratio, axis=1)


# Define the formula to calculate Quick Ratio and multiply by 100
def calculate_quick_ratio(row):
    if pd.isnull(row['Quick Ratio']):
        total_current_liabilities = row['Total Current Liabilities']
        if total_current_liabilities != 0:
            return ((row['Total Current Assets'] - row['Inventory']) / total_current_liabilities) * 100
        else:
            return 0  # Handle division by zero case
    else:
        return row['Quick Ratio']
    
    

# Apply the formula to fill null values in the 'Quick Ratio' column
fundamentals['Quick Ratio'] = fundamentals.apply(calculate_quick_ratio, axis=1)


# Replace null values in 'Estimated Shares Outstanding' with 0
fundamentals['Estimated Shares Outstanding'].fillna(0, inplace=True)




# Define the formula to calculate Earnings Per Share (EPS)
def calculate_eps(row):
    net_income = row['Net Income']
    estimated_shares_outstanding = row['Estimated Shares Outstanding']
    
    if pd.isnull(row['Earnings Per Share']):
        if estimated_shares_outstanding != 0:
            return net_income / estimated_shares_outstanding
        else:
            return 0  # Set EPS to 0 if Estimated Shares Outstanding is 0
    else:
        return row['Earnings Per Share']

# Apply the formula to fill null values in the 'Earnings Per Share' column
fundamentals['Earnings Per Share'] = fundamentals.apply(calculate_eps, axis=1)

# Replace empty values in 'For Year' with values from 'Period Ending'
fundamentals['For Year'].fillna(fundamentals['Period Ending'], inplace=True)


st.write("Null values for fundamentals")
st.dataframe(fundamentals.isnull().sum().reset_index())

#need latest price of every company share in fundamentals dataset and make new columns of MArket value of EQ
fundamentals.rename(columns={'Ticker Symbol': 'symbol'}, inplace=True)
fundamentals = pd.merge(fundamentals, latest_prices[['symbol', 'close']], on='symbol', how='left')

fundamentals['Market Value of Equity'] = fundamentals['close'] * fundamentals['Estimated Shares Outstanding']

st.subheader('Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E')
st.write('A = Working Capital / Total Assets')
st.write('B = Retained Earnings / Total Assets')
st.write('C = Earnings Before Interest and Tax / Total Assets')
st.write('D = Market Value of Equity / Total Liabilities')
st.write('E = Sales / Total Assets')

# Calculate the components A, B, C, D, and E
fundamentals['A'] = (fundamentals['Total Current Assets'] - fundamentals['Total Current Liabilities']) / fundamentals['Total Assets']
fundamentals['B'] = fundamentals['Retained Earnings'] / fundamentals['Total Assets']
fundamentals['C'] = fundamentals['Earnings Before Interest and Tax'] / fundamentals['Total Assets']
fundamentals['D'] = fundamentals['Market Value of Equity'] / fundamentals['Total Liabilities']
fundamentals['E'] = fundamentals['Sales, General and Admin.'] / fundamentals['Total Assets']

# Calculate the Z-Score using the formula Z = 1.2A + 1.4B + 3.3C + 0.6D + 1.0E
fundamentals['Z-Score'] = 1.2 * fundamentals['A'] + 1.4 * fundamentals['B'] + 3.3 * fundamentals['C'] + 0.6 * fundamentals['D'] + 1.0 * fundamentals['E']



# Convert 'Period Ending' to datetime for correct sorting
fundamentals['Period Ending'] = pd.to_datetime(fundamentals['Period Ending'])

# Find the index of the latest date for each 'symbol'
latest_data_index = fundamentals.groupby('symbol')['Period Ending'].idxmax()

# Select the rows with the latest date for each 'symbol'
latest_data = fundamentals.loc[latest_data_index]

# Define function to categorize Z-Score
def categorize_z_score(z_score):
    if 0 <= z_score <= 1.8:
        return 'Company may declare bankruptcy'
    elif 1.8 < z_score <= 3:
        return 'Company is likely to declare bankruptcy'
    else:
        return 'Company will not declare bankruptcy'

# Apply the categorize function to create a new column 'Bankruptcy Prediction'
latest_data['Bankruptcy Prediction'] = latest_data['Z-Score'].apply(categorize_z_score)


# Add filter options
prediction_filter = st.selectbox('Filter by Bankruptcy Prediction', ['Company may declare bankruptcy', 'Company is likely to declare bankruptcy', 'Company will not declare bankruptcy'])

# Filter the DataFrame based on the selected category
filtered_data = latest_data[latest_data['Bankruptcy Prediction'] == prediction_filter]

# Display the Z-Score and the new column
st.dataframe(filtered_data[['For Year', 'symbol', 'Z-Score', 'Bankruptcy Prediction']])

###########################################################################################

st.title("Ratios")
# Interest Coverage Ratio
fundamentals['Interest Coverage Ratio'] = fundamentals['Earnings Before Interest and Tax'] / fundamentals['Earnings Before Tax']

# Quick Ratio
fundamentals['Quick Ratio data'] = fundamentals['Quick Ratio'] / 100

# Debt-Equity Ratio
fundamentals['Debt-Equity Ratio'] = (fundamentals['Short-Term Debt / Current Portion of Long-Term Debt'] + fundamentals['Long-Term Debt']) / fundamentals['Total Assets']

# Net Cash Flow-Operating
net_cash_flow_operating = fundamentals[['For Year', 'symbol', 'Net Cash Flow-Operating']]

# Display Sidebar Options
selected_stock = st.sidebar.selectbox('Select Stock For data', fundamentals['symbol'].unique())

# Filter DataFrame based on selected stock
selected_stock_data = fundamentals[fundamentals['symbol'] == selected_stock]
col1, col2 = st.columns([1, 1])
with col1:
# Display Interest Coverage Ratio
 st.sidebar.subheader('Interest Coverage Ratio')
 st.write("Interest Coverage Ratio")
 st.write(selected_stock_data[['For Year', 'symbol', 'Interest Coverage Ratio']])
with col2:
# Display Quick Ratio
 st.sidebar.subheader('Quick Ratio')
 st.write("Quick Ratio data")
 st.write(selected_stock_data[['For Year', 'symbol', 'Quick Ratio data']])
col1, col2 = st.columns([1, 1])
with col1:
# Display Debt-Equity Ratio
 st.sidebar.subheader('Debt-Equity Ratio')
 st.write("Debt-Equity Ratio")
 st.write(selected_stock_data[['For Year', 'symbol', 'Debt-Equity Ratio']])
with col2:
# Display Net Cash Flow-Operating

 st.sidebar.subheader('Net Cash Flow-Operating')
 st.write("Net Cash Flow-Operating")
 st.write(selected_stock_data[['For Year', 'symbol', 'Net Cash Flow-Operating']])

st.info('The interest coverage ratio is used to measure how well a firm can pay the interest due on outstanding debt.Therefore, a higher interest coverage ratio indicates stronger financial health.Analysts generally look for ratios of at least two (2) while three (3) or more is preferred.')
st.info('The quick ratio measures a companys ability to quickly convert liquid assets into cash to pay for its short-term financial obligations. A positive quick ratio can indicate the companys ability to survive emergencies or other events that create temporary cash flow problems. In general, analysts believe if the ratio is more than 1.0, a business can pay its immediate expenses. If it is less than 1.0, it cannot.')
st.info('The debt to equity ratio is a measure of a companys financial leverage, and it represents the amount of debt and equity being used to finance a companys assets.A good debt to equity ratio is around 1 to 1.5.A high debt to equity ratio indicates a business uses debt to finance its growth')
st.info('Positive net cash flow (above 0) is generally a sign of financial soundness and good management')

st.write("References")
st.write("https://www.carboncollective.co/sustainable-investing/z-score#:~:text=company's%20fiscal%20state.-,Z%2DScore%20Analysis,buying%20or%20selling%20an%20investment")
st.write("https://www.kaggle.com/datasets/dgawlik/nyse")



