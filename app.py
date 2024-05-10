import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np


st.set_page_config(page_title='DoorDash Data Analysis', layout='wide')

# Load and preprocess data
@st.cache_data
def load_data():
    data = pd.read_csv('doordash.csv')
    data.fillna(0, inplace=True)
    return data

data = load_data()

# Explode the 'cuisines' to count occurrences
data['cuisines'] = data['cuisines'].str.split('|')
data = data.explode('cuisines')

# Page Configuration


# Sidebar for choosing the analysis
st.sidebar.title("Analysis Type")

# List of analysis types
analysis_types = [
    "Geographic Distribution of Restaurants",
    "Miscellaneous Insights",
    "Show popular restuarants",
    "Cuisine Popularity",
    "Review Analysis",
    "Average Delivery Time by City",
    "Customer Segmentation",
    "Top Specialty Items by City",
    "Cuisine Recommendation System",
]


# Allow the user to select an analysis type
choice = st.sidebar.radio("Choose the Analysis", analysis_types)

import plotly.express as px

if choice == "Geographic Distribution of Restaurants":
    st.title("Geographic Distribution of Restaurants")
    # Ensure the data has the necessary columns
    if 'latitude' in data.columns and 'longitude' in data.columns and 'loc_name' in data.columns:
        # Create a scatter mapbox plot using Plotly
        fig = px.scatter_mapbox(data,
                                lat="latitude",
                                lon="longitude",
                                hover_name="loc_name",
                                color_discrete_sequence=["fuchsia"],  # You can customize the point color
                                zoom=3,
                                height=300)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig)
    else:
        st.write("Data does not contain the required columns.")



elif choice == "Cuisine Popularity":
    st.title("Cuisine Popularity globally")
    # Filter out 'Miscellaneous' before counting
    filtered_data = data[data['cuisines'] != 'Miscellaneous']
    # Count of each cuisine type
    cuisine_count = filtered_data['cuisines'].value_counts().reset_index()
    cuisine_count.columns = ['Cuisine', 'Count']
    fig = px.bar(cuisine_count.head(20), x='Cuisine', y='Count', title='Top 20 Popular Cuisines')
    st.plotly_chart(fig)


    st.title("Cuisine Popularity by State")
    state_cuisine_count = data.groupby(['searched_state', 'cuisines']).size().reset_index(name='Count')
    state_cuisine_count = state_cuisine_count[state_cuisine_count['cuisines'] != 'Miscellaneous']
    state_cuisine_count.sort_values('Count', ascending=False, inplace=True)
    fig = px.bar(state_cuisine_count, x='searched_state', y='Count', color='cuisines', title="Most Popular Cuisines by State")
    st.plotly_chart(fig)

    st.title("Cuisine Popularity by City")
    
    # Assuming 'cuisines' column has multiple cuisines separated by '|'
    # Explode the 'cuisines' column to create a row for each cuisine per city
    data_expanded = data.assign(cuisines=data['cuisines'].str.split('|')).explode('cuisines')
    
    # Remove 'Miscellaneous' entries if they exist
    data_expanded = data_expanded[data_expanded['cuisines'] != 'Miscellaneous']
    
    # Group by 'searched_city' and 'cuisines' and sum 'review_count'
    cuisine_popularity = data_expanded.groupby(['searched_city', 'cuisines']).agg({'review_count': 'sum'}).reset_index()
    
    # Sorting values to get better visualization insights
    cuisine_popularity.sort_values(by=['searched_city', 'review_count'], ascending=[True, False], inplace=True)
    
    # Creating a bar plot with Plotly Express
    fig = px.bar(cuisine_popularity, x='searched_city', y='review_count', color='cuisines', 
                 title="Cuisine Popularity by City Based on Review Count",
                 labels={"review_count": "Total Review Count", "cuisines": "Cuisine", "searched_city": "City"})
    
    # Showing the figure in the Streamlit app
    st.plotly_chart(fig)


elif choice == "Review Analysis":
    st.title("Review Rating Analysis")
    
    # Distribution of review ratings
    fig = px.histogram(data, x='review_rating', nbins=50, title='Distribution of Review Ratings')
    st.plotly_chart(fig)

    # Split 'cuisines', explode, and filter out 'Miscellaneous'
    data['cuisines'] = data['cuisines'].str.split('|')
    exploded_data = data.explode('cuisines')
    exploded_data = exploded_data[exploded_data['cuisines'] != 'Miscellaneous']

    # Calculate average rating and review count by cuisine
    avg_rating_cuisine = exploded_data.groupby('cuisines').agg(
        average_rating=pd.NamedAgg(column='review_rating', aggfunc='mean'),
        review_count=pd.NamedAgg(column='review_rating', aggfunc='count')
    ).reset_index()

    # Filter to include only cuisines with a significant number of reviews
    avg_rating_cuisine = avg_rating_cuisine[avg_rating_cuisine['review_count'] > 50]  # Adjust the threshold as needed
    avg_rating_cuisine.sort_values('average_rating', ascending=False, inplace=True)

    fig = px.bar(avg_rating_cuisine, x='cuisines', y='average_rating', title='Average Review Rating per Cuisine')
    st.plotly_chart(fig)

    rating_pivot = exploded_data.pivot_table(
        index='cuisines',
        columns='searched_city',
        values='review_rating',
        aggfunc='mean',
        fill_value=0,
        
    )
    # Heatmap of Review Ratings by Cuisine and City
    fig = px.imshow(rating_pivot, aspect="auto", title="Average Review Ratings by Cuisine and City", height=600, color_continuous_scale=[(0, 'lightblue'), (1, 'darkblue')] )

    fig.update_layout(
        coloraxis_colorbar=dict(
            title='Average Ratings', 
            ticks='outside',
            tickvals=[rating_pivot.min().min(), rating_pivot.max().max()]
        )
    )

    st.plotly_chart(fig)


elif choice == "Average Delivery Time by City":
    st.title("Average Delivery Time by City")
    city_delivery_time = data.groupby('searched_city')['delivery_time'].mean().sort_values().head(10).reset_index()
    city_delivery_time.columns = ['City', 'Average Delivery Time']
    fig = px.bar(city_delivery_time, x='City', y='Average Delivery Time', title="Average Delivery Time in Top 10 Cities")
    st.plotly_chart(fig)

elif choice == "Customer Segmentation":
    st.title("Customer Segmentation based on Review Count and Rating")
    # Cluster users based on review count and rating
    X = data[['review_count', 'review_rating']].dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=4, random_state=0).fit(X_scaled)
    cluster_labels = kmeans.labels_
    
    # Add clusters to original data
    X_clustered = pd.DataFrame(X_scaled, columns=['review_count', 'review_rating'])
    X_clustered['cluster'] = cluster_labels
    fig = px.scatter(X_clustered, x='review_count', y='review_rating', color='cluster')
    st.plotly_chart(fig)

elif choice == "Top Specialty Items by City":
    st.title("Top Specialty Items by City")
    
    # Explode the 'Specialty Items' column into separate rows
    data['Specialty Items'] = data['Specialty Items'].str.split('|')
    exploded_data = data.explode('Specialty Items')

    # Remove rows with 'Miscellaneous' in 'Specialty Items'
    filtered_data = exploded_data[exploded_data['Specialty Items'] != 'Miscellaneous']

    # Group by city and specialty item, then count occurrences
    specialty_count = filtered_data.groupby(['searched_city', 'Specialty Items']).size().reset_index(name='Count')
    specialty_count.sort_values('Count', ascending=False, inplace=True)

    # Plot the top 20 results
    fig = px.bar(specialty_count.head(20), x='searched_city', y='Count', color='Specialty Items', title="Top Specialty Items by City")
    st.plotly_chart(fig)


elif choice == "Generic Insights":
    st.title("Generic Insights")
    # Distance vs Delivery Time
    fig = px.scatter(data, x='distance', y='delivery_time', title='Distance vs. Delivery Time', trendline="ols")
    st.plotly_chart(fig)

    fig = px.scatter(data, x='review_count', y='review_rating', title="Review Count vs. Customer Rating", trendline="ols")
    st.plotly_chart(fig)

    st.set_option('deprecation.showPyplotGlobalUse', False)
    # Correlation heatmap
    st.write("Correlation Matrix")
    corr = data[['review_count', 'review_rating', 'delivery_time']].corr()
    sns.heatmap(corr, annot=True)
    st.pyplot()


elif choice == "Show popular restuarants":
    # Function to clean and explode complex columns
    def clean_and_explode(data, column):    
        # Remove 'Miscellaneous', explode the column, and drop duplicates
        data = data.dropna(subset=[column])
        data[column] = data[column].apply(lambda x: x.split('|'))
        data = data.explode(column)
        data = data[data[column] != 'Miscellaneous']
        return data


    # Clean 'cuisines', 'Specialty Items', 'Dietary Preferences'
    data = clean_and_explode(data, 'cuisines')

    # Main app
    st.title("Show popular restuarants")
    # Assuming data preprocessing is done elsewhere in the code
    def normalize_column(data, column_name):
        max_value = data[column_name].max()
        return data[column_name] / max_value

    # Weighted score calculation
    def calculate_weighted_score(data, rating_weight=0.7, count_weight=0.3):
        # Normalize columns
        data['normalized_rating'] = normalize_column(data, 'review_rating')
        data['normalized_count'] = normalize_column(data, 'review_count')

        # Calculate weighted score
        data['weighted_score'] = (data['normalized_rating'] * rating_weight) + (data['normalized_count'] * count_weight)
        return data

    # Apply the function
    data = calculate_weighted_score(data)
    # Independent selections
    all_cuisines = sorted(data['cuisines'].unique())
    all_cities = sorted(data['searched_city'].unique())
    all_restaurants = sorted(data['loc_name'].unique())

    
    selected_cities = st.multiselect('Select Cities', all_cities)
    selected_cuisines = st.multiselect('Select Cuisines', all_cuisines)
    selected_restaurants = st.multiselect('Select Restaurants', all_restaurants)

    # Filter data based on selections
    filtered_data = data.copy()
    if selected_cities:
        filtered_data = filtered_data[filtered_data['searched_city'].isin(selected_cities)]
    if selected_cuisines:
        filtered_data = filtered_data[filtered_data['cuisines'].isin(selected_cuisines)]
    if selected_restaurants:
        filtered_data = filtered_data[filtered_data['loc_name'].isin(selected_restaurants)]

    # Sort by review_rating and review_count to handle ties, then pick the top 10
    filtered_data = filtered_data.sort_values(
        by='weighted_score', ascending=False
        ).drop_duplicates(subset='address').head(10)

    # Ensure that filtered data is not empty
    if not filtered_data.empty:
        # Display results
        st.write("Top 10 Restaurants by City, Cuisine, and Restaurant Name:")
        st.dataframe(filtered_data[['searched_city', 'loc_name', 'cuisines', 'review_rating', 'review_count', 'delivery_time', 'address']])    
    else:
        st.write("No data matches your selections.")
    

elif choice == "Cuisine Recommendation System":
    # Load the recommendations data
    data = pd.read_csv('recommendations.csv')

    # Title for the section
    st.title("Cuisine Recommendation by City")

    # Create a multi-select dropdown menu for cities
    selected_cities = st.multiselect("Select Cities", options=data['searched_city'].unique(), default=data['searched_city'].unique()[0])

    # Display recommendations for each selected city
    if selected_cities:
        # Filter data for selected cities
        selected_data = data[data['searched_city'].isin(selected_cities)]
        display_columns = ['searched_city', 'Predicted Cuisine', 'Predicted Specialty Item', 'Predicted Meal Type', 'Predicted Dietary Preference']
        # Display the filtered data in a table
        st.table(selected_data[display_columns])

