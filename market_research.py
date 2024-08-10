import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import requests

# Function to get census data
def get_census_data(census_year_list, state_code_num, features, geographic_level='tract'):
    census_api_key = "6e4c8a9351f80033625058c6e007b6b2580ce489"
    # Census Dictionary
    census_d = {
        'population': {'B01001_001E': 'total_population'},
        'income': {'B19013_001E': 'median_household_income'},
        'employment': {'B23025_001E': 'total_in_labor_workforce', 'B23025_007E': 'total_not_in_labor_workforce'},
        'home_estimate': {'B25077_001E': 'median_home_value'},
        'rent_estimate': {'B25031_001E': 'gross_rent'},
        'education': {
            'B06009_001E': 'total_education',
            'B06009_002E': 'education_no_hs',
            'B06009_003E': 'education_hs',
            'B06009_004E': 'education_associates',
            'B06009_005E': 'education_bachelors',
            'B06009_006E': 'education_graduate'
        },
        'occupancy': {'B25002_001E': 'total_occupancy_status', 'B25002_002E': 'occupancy_status_occupied', 'B25002_003E': 'occupancy_status_vacant'},
        'tenure': {'B25003_001E': 'total_home_tenure', 'B25003_002E': 'home_tenure_owner_occupied', 'B25003_003E': 'home_tenure_renter_occupied'},
        'new_builds': {
            'B25035_001E': 'median_construction_year',
            'B25034_001E': 'total_housing_units',
            'B25034_002E': 'housing_units_2020+',
            'B25034_003E': 'housing_units_2010-2019'
        }
    }

    # Construct Dictionary
    input_census_keys_d = {}
    for feature in features:
            input_census_keys_d.update(census_d[feature])

    # Retrieve Data
    code_list = ",".join(list(input_census_keys_d.keys()))
    df_list = []

    for census_year in census_year_list:
        # URL
        if geographic_level == 'state':
            census_url = f'https://api.census.gov/data/{census_year}/acs/acs5?get=NAME,STATE,{code_list}&for=state:{state_code_num}&key={census_api_key}'
        elif geographic_level == 'tract':
            census_url =  f'https://api.census.gov/data/{census_year}/acs/acs5?get=NAME,{code_list}&for=tract:*&in=state:{state_code_num}&in=county:*&key={census_api_key}'
        
        # Request
        _response = requests.get(census_url)
        df = pd.DataFrame(_response.json())

        # Modify dataframe for readability
        df.columns = df.iloc[0]
        df = df[1:]
        df['reporting_date'] = census_year

        cols = []
        for c in df.columns:
            if c in input_census_keys_d:
                df[c] = df[c].astype('Int64')
                cols.append(input_census_keys_d[c])
            else:
                cols.append(c)
        df.columns = cols

        df_list.append(df)

    df_all = pd.concat(df_list)

    if geographic_level == 'state':
        df_all = df_all.drop(columns=[geographic_level])
        df_all.columns = [c.lower() for c in df_all.columns]
        df_all['region_type'] = geographic_level
        df_all['geo_id'] = None
    elif  geographic_level == 'tract':
        df_all.columns = [c.lower() for c in df_all.columns]
        df_all['region_type'] = geographic_level
        cols = list(df_all)
        cols.insert(1, cols.pop(cols.index('state')))
        df_all = df_all.loc[:, cols]
        df_all['geo_id'] = df_all.apply(lambda x: 'GEOID ' + x['state'] + x['county'] + x['tract'], axis=1)
        df_all = df_all.drop(columns=['county', 'tract'])
    
    return df_all

# Streamlit App
st.title("Real Estate Investment Analysis")
st.write("""
This app helps you determine the best markets to invest in based on factors such as population growth, 
higher education growth, labor workforce growth, and occupancy percentage growth.
""")

# User Input
st.sidebar.header("User Input")

# Restrict state code to numbers 1 to 56
state_code = st.sidebar.number_input("Enter State Code", min_value=1, max_value=56) # Sidebar link

if state_code < 10:
    state_code = f'0{state_code}'
else:
    state_code = str(state_code)

st.sidebar.markdown(
    """
    For a list of state codes, please visit the following [website](https://www.census.gov/library/reference/code-lists/ansi/ansi-codes-for-states.html).
    """,
    unsafe_allow_html=True
)
years = ['2020', '2022']
features = ['population', 'occupancy', 'education', 'employment', 'home_estimate', 'rent_estimate']  # Hardcoded features

if state_code and years:
    # Load data using the user input
    data = get_census_data(years, state_code, features, 'tract')

    # Filter the data for the selected years
    df_filtered = data[data['reporting_date'].isin(['2020', '2022'])]

    # Ensure geo_id is included
    df_filtered.set_index('geo_id', inplace=True)

    # **Data Validation: Remove rows with 0 or less in critical features**
    critical_features = ['total_population', 'total_in_labor_workforce', 'occupancy_status_occupied', 'total_occupancy_status', 'median_home_value', 'gross_rent']
    for feature in critical_features:
        df_filtered = df_filtered[df_filtered[feature] > 0]

    # Separate the data by year
    df_2020 = df_filtered[df_filtered['reporting_date'] == '2020']
    df_2022 = df_filtered[df_filtered['reporting_date'] == '2022']

    # **Remove non-matching rows based on geo_id**
    matching_geo_ids = df_2020.index.intersection(df_2022.index)
    df_2020 = df_2020.loc[matching_geo_ids]
    df_2022 = df_2022.loc[matching_geo_ids]

    # Calculate the change from 2020 to 2022
    df_change = df_2022.copy()
    df_change['population_change'] = df_2022['total_population'] - df_2020['total_population']
    df_change['labor_workforce_change'] = df_2022['total_in_labor_workforce'] - df_2020['total_in_labor_workforce']
    df_change['occupancy_pct_change'] = (
        (df_2022['occupancy_status_occupied'] / df_2022['total_occupancy_status']) * 100 -
        (df_2020['occupancy_status_occupied'] / df_2020['total_occupancy_status']) * 100
    )

    # Calculate the change in higher education levels
    df_change['higher_education_change'] = (
        (df_2022['education_associates'] + df_2022['education_bachelors'] + df_2022['education_graduate']) -
        (df_2020['education_associates'] + df_2020['education_bachelors'] + df_2020['education_graduate'])
    )
    
    # Calculate cash flow 
    df_change['cash_flow'] = df_2022['gross_rent'] / df_2022['median_home_value']

    # Normalize the data
    scaler = MinMaxScaler()

    df_change[['population_change_norm', 'higher_education_change_norm', 'labor_workforce_change_norm', 'occupancy_pct_change_norm', 'cash_flow_norm']] = scaler.fit_transform(
        df_change[['population_change', 'higher_education_change', 'labor_workforce_change', 'occupancy_pct_change', 'cash_flow']]
    )

    # Weight the factors
    st.sidebar.header("Weighting Factors")
    population_weight = st.sidebar.slider("Population Change Weight", 0.0, 1.0, 0.4)
    education_weight = st.sidebar.slider("Higher Education Change Weight", 0.0, 1.0, 0.1)
    labor_weight = st.sidebar.slider("Labor Workforce Change Weight", 0.0, 1.0, 0.2)
    occupancy_weight = st.sidebar.slider("Occupancy Percentage Change Weight", 0.0, 1.0, 0.2)
    cash_flow_weight = st.sidebar.slider("Cash Flow Weight", 0.0, 1.0, 0.1)

    total_weight = round(population_weight + education_weight + labor_weight + occupancy_weight + cash_flow_weight, 2)

    if total_weight != 1.0:
        st.warning(f"Current weights do not sum to 1.0 (Sum = {total_weight:.2f}). Adjust the weights so that they sum to 1.0.")
    else:
        df_change['investment_score'] = (
            population_weight * df_change['population_change_norm'] +
            education_weight * df_change['higher_education_change_norm'] +
            labor_weight * df_change['labor_workforce_change_norm'] +
            occupancy_weight * df_change['occupancy_pct_change_norm'] +
            cash_flow_weight * df_change['cash_flow_norm']
        )

        # Rank the Census Tracts
        df_change['investment_rank'] = df_change['investment_score'].rank(ascending=False)

        # Display the top-ranked markets
        st.subheader("Top Markets to Invest In")
        top_n = st.slider("Number of Top Markets to Display", 1, 20, 10)
        best_markets = df_change.sort_values('investment_rank').head(top_n)

        # Reset the index to include geo_id as a column
        best_markets.reset_index(inplace=True)

        st.write(best_markets[['name', 'geo_id', 'investment_score', 'investment_rank']])

        # Optional: Download the results
        st.download_button(
            label="Download Results as CSV",
            data=best_markets.to_csv(index=False).encode('utf-8'),
            file_name='best_markets.csv',
            mime='text/csv'
    )
else:
    st.write("Please enter a state code and select at least one year to proceed.")
