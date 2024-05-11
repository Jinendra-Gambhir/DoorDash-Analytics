
# DoorDash-Analytics

This project uses Python, PySpark, and data visualization libraries to analyze DoorDash operational data. It provides insights into cuisine popularity, delivery efficiency, and customer preferences across various regions. The analytics suite includes predictive modeling for delivery times and a recommendation system for cuisines and specialty items, optimizing business strategies and enhancing customer satisfaction. Interactive visualizations created with Seaborn, Matplotlib, and Plotly enhance the interpretability of complex data, supporting strategic decisions to boost service quality and market presence. Using Streamlit, the project also provides an interactive web application that allows users to explore various metrics such as delivery times, customer segmentation, and review analyses.



## Acknowledgements

 - [California State University Fullerton](https://www.fullerton.edu/)
 - [Kaggle](https://www.kaggle.com/datasets/polartech/doordash-restaurant-data)

## Tech Stack

* **Pyspark:** Framework for handling big data processing in Python, used for reading and processing large datasets efficiently.
* **Python:** Primary programming language used for data manipulation and analysis.
* **Pandas:** Library for data manipulation and analysis, used extensively for operations like reading CSV files, filtering, and aggregations.
* **NumPy:** Utilized for numerical operations within the data preprocessing and analysis steps.
* **Plotly Express:** Employed for creating interactive charts and visualizations.
* **Tableau:** Utilized for data visualization, offering interactive dashboards and insightful analytics, enhancing decision-making in the project.
* **Streamlit:** Framework for building interactive web applications directly from Python scripts.
* **Seaborn and Matplotlib:** Used for additional statistical visualizations, primarily within exploratory data analysis.
* **Scikit-learn:** Provides tools for data preprocessing (like StandardScaler) and machine learning (like KMeans for clustering).
* **Google Colab:** Platforms for writing and executing Python code in an interactive environment, used for script development and testing.

## Data Columns
1. searched_zipcode: ZIP codes for restaurant locations.
2. searched_lat: Latitude coordinates for search queries.
3. searched_lng: Longitude coordinates for search queries.
4. searched_address: Full addresses where searches occurred.
5. searched_state: State names from search data.
6. searched_city: City names from search data.
7. searched_metro: Metro areas from searches.
8. latitude: Restaurant latitude coordinates.
9. longitude: Restaurant longitude coordinates.
10. distance: Distance to restaurant in kilometers.
11. loc_name: Names of the restaurants.
12. loc_number: Numerical identifiers for restaurants.
13. address: Physical addresses of restaurants.
14. cuisines: Types of cuisines offered.
15. delivery_time: Time taken for delivery in minutes.
16. review_count: Number of reviews for each restaurant.
17. review_rating: Average rating out of 5.
18. Specialty Items: Specific dishes offered.
19. Meal Types: Categories like breakfast, lunch.
20. Dietary Preferences: Diet-specific options available.
  
## Features

* **Cuisine Popularity Analysis:** Identifies the most popular cuisines in various cities and states using PySpark to process large datasets efficiently.
* **Review Rating Insights:** Computes average review ratings for each cuisine, providing data to gauge customer satisfaction and preference trends.
* **Delivery Time Analysis:** Investigates the correlation between delivery times and review ratings, exploring how delivery performance impacts customer feedback.
* **Spatial Analysis:** Analyzes the relationship between distance and delivery time, helping identify how geographical factors affect delivery efficiency.
* **Customer Segmentation:** Uses K-Means clustering to segment customers based on review counts and ratings, enhancing targeted marketing strategies.
* **Predictive Modeling for Delivery Times:** Develops a linear regression model to predict delivery times based on distance and location, optimizing route planning and operational efficiency.
* **Cuisine Recommendation System:** Implements an ALS-based recommendation system to suggest cuisines and specialty items to users based on their location and past preferences, personalized to enhance user experience.
* **Trend Analysis:** Examines time-related data to understand peak periods and potential seasonal variations in delivery and order patterns.
* **Customer Satisfaction Metrics:** Explores additional metrics like sentiment analysis from customer feedback, providing a deeper understanding of the overall service impact on customers.
* **Interactive Visualizations:** Utilizes libraries like Matplotlib, Seaborn, and Plotly for dynamic and interactive data visualizations, making the analytics accessible and understandable for strategic decision-making.


# Usage
## Exploring Cuisines and Counting by City
- This snippet explodes the 'cuisines' column to count the occurrences of each cuisine type in different cities, helping to identify popular tastes in each area.

```PySpark
from pyspark.sql.functions import explode, split, col

df = spark.read.csv(data_path, header=True, inferSchema=True)
df = df.select("searched_city", "cuisines")
df = df.withColumn("cuisine", explode(split(col("cuisines"), "\\|")))
cuisine_counts = df.groupBy("searched_city", "cuisine").count()
cuisine_counts.show()
```
## Customer Segmentation Using K-Means Clustering
- This snippet uses K-Means clustering to segment customers based on review count and review ratings, which can help in targeted marketing and service improvement.
```PySpark
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('/content/cleaned_doordash.csv')
features = ['review_count', 'review_rating']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])
kmeans = KMeans(n_clusters=3, random_state=42).fit(data_scaled)
data['cluster'] = kmeans.labels_
```


## Screenshots

![App Screenshot](https://github.com/Jinendra-Gambhir/DoorDash-Analytics/blob/main/Tableau/Average%20Delivery%20Time%20by%20City.png)
![App Screenshot](https://github.com/Jinendra-Gambhir/DoorDash-Analytics/blob/main/Tableau/City%20Comparison.png)
![App Screenshot](https://github.com/Jinendra-Gambhir/DoorDash-Analytics/blob/main/Tableau/Popular%20cuisines%20by%20city.png)


## Deployment

- [Streamlit](https://doordash-analytics.streamlit.app/)

## Authors

- [Jinendra Gambhir](https://www.github.com/Jinendra-Gambhir)

- [Jayraj Arora](https://github.com/JAYRAJARORA)


## Feedback

Feedback is welcome! Feel free to open an issue for suggestions or bug reports.


## 🔗 Links
[![GitHub](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](github.com/Jinendra-Gambhir/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/jinendragambhir/)

## License

[MIT](https://choosealicense.com/licenses/mit/)

