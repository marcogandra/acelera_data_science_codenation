import streamlit as st
import pandas as pd
import numpy as np

st.title("Uber pickups in NYC")

date_column = "date/time"
data_url = ("https://s3-us-west-2.amazonaws.com/"
            "streamlit-demo-data/uber-raw-data-sep14.csv.gz")
            
@st.cache
def load_data(nrows):
    ## Loads the data, puts it in a pandas dataframe
    ## and converts the date column from text to
    ## datetime
    data = pd.read_csv(data_url, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis="columns", inplace=True)
    data[date_column] = pd.to_datetime(data[date_column])
    return(data)    

# create a text element and let the reader know the data is loading
data_load_state = st.text("The data is loading...")

# Load 10,000 rows of data into the dataframe
data = load_data(10000)

# Notify the reader that the data was successfully loaded
data_load_state.text("The data was loaded!")

# the raw data
if st.checkbox("Show raw data"):
    st.subheader("Raw data")
    st.write(data)

# histogram
st.subheader("Number of pickups by hour")
hist_values = np.histogram(data[date_column].dt.hour, bins=24, range=(0,24))[0]
st.bar_chart(hist_values)

# the data on a map
hour_to_filter = st.slider("Choose an hour:", 0, 23, 17)
filtered_data = data[data[date_column].dt.hour == hour_to_filter]
st.subheader(f"Map of all pickups at {hour_to_filter}:00")
st.map(filtered_data)