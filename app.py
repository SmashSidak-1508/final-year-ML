from reportlab.lib import colors
import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import plotly.express as px

# Loading the model
filename = "dietrec.sav"
loaded_model = pickle.load(open(filename, 'rb'))

# Reading Food.csv for diet recommendation
data = pd.read_csv("food.csv")
diet = data.drop("Food_items", axis=1)

# Handle missing values using SimpleImputer
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
diet_imputed = imputer.fit_transform(diet)

# Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
diet_scaled = scaler.fit_transform(diet_imputed)

# Apply KMeans clustering
km = KMeans(n_clusters=4)
y_predicted = km.fit_predict(diet_scaled)

# Adding a new column containing the cluster a food item is part of
data['cluster'] = y_predicted

# Streamlit app
# st.set_page_config(page_title="DIET RECOMMENDATION SYSTEM", page_icon="diet.ico", layout="wide", initial_sidebar_state="expanded")

# ... (remaining code as before)

# Streamlit app
st.set_page_config(page_title="DIET RECOMMENDATION SYSTEM", page_icon="diet.ico",layout="wide",  # Optional: set the layout to wide
    initial_sidebar_state="expanded",
    )
st.markdown(
    """
    <style>
    body {
        background-color: #FFFFFF;  /* Set background color to white */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("DIET RECOMMENDATION SYSTEM")

# User input
name = st.text_input("Enter your name")
age = st.number_input("Enter your age", min_value=0, step=1)

weight = st.number_input("Enter your weight in kg", min_value=0.0, step=0.1, format="%.1f")
height = st.number_input("Enter your height in m", min_value=0.01, step=0.01, format="%.2f")  # Set a minimum height value

# Check for zero height
if height == 0:
    st.error("Error: Height should be a non-zero value.")
else:
    # Calculate BMI
    bmi = weight / (height ** 2)

    # Determine weight category
    if bmi < 18.5:
        weight_category = "Underweight"
    elif bmi >= 18.5 and bmi < 25:
        weight_category = "Normal weight"
    else:
        weight_category = "Overweight"

    # Predict using the model
    Xdata = {'Height': [height], 'Weight': [weight], 'BMI': [bmi]}
    df = pd.DataFrame(Xdata, columns=['Height', 'Weight', 'BMI'])
    predicted = loaded_model.predict(df)
    predicted_cluster = predicted[0]

    # Display BMI and weight category
    st.markdown(f"### Your BMI is **{bmi:.2f}** - **{weight_category}**", unsafe_allow_html=True)

    st.markdown("### Diet Recommendation:")
    recommended_food = data[data['cluster'] == predicted_cluster].Food_items.sample(21, replace=True).values

    # Create a DataFrame for day-wise schedule
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    schedule = pd.DataFrame(columns=['Breakfast', 'Lunch', 'Dinner'], index=days)

    # Populate the schedule DataFrame with recommended foods
    index = 0
    for day in days:
        schedule.loc[day, 'Breakfast'] = recommended_food[index]
        schedule.loc[day, 'Lunch'] = recommended_food[index + 1]
        schedule.loc[day, 'Dinner'] = recommended_food[index + 2]
        index += 3

    # Display the schedule as a table with enhanced styling
    st.table(schedule)
    st.subheader("Visualizations")

    fig_height_weight = px.scatter(data, x='Food_items', y='Calories', color='cluster', title='Food item vs Calorie')
    st.plotly_chart(fig_height_weight)


    fig_bmi_calories = px.scatter(data, x='Food_items', y='Sugars', color='cluster', title='Food items vs Sugars')
    st.plotly_chart(fig_bmi_calories)


    fig_age_calories = px.scatter(data, x='Food_items', y='Fibre', color='cluster', title='Food items vs Fibre')
    st.plotly_chart(fig_age_calories)
    
    recommended_food = data[data['cluster'] == predicted_cluster].Food_items.sample(21, replace=True).values
    avg_calories_recommended = data[data['Food_items'].isin(recommended_food)].groupby('Food_items')['Calories'].mean().reset_index()

    fig_food_calories = px.bar(avg_calories_recommended, x='Food_items', y='Calories', title='Recommended Food vs Calories')
    fig_food_calories.update_xaxes(title='Recommended Food')
    fig_food_calories.update_yaxes(title='Calories')
    fig_food_calories.update_layout(xaxis_tickangle=-45)  # Rotate x-axis labels for better readability
    st.plotly_chart(fig_food_calories)
    # Calculate average calories per cluster
    avg_calories = data.groupby('cluster')['Calories'].mean().reset_index()

    # Visualize average calorie count per cluster using Plotly bar chart
    fig = px.bar(avg_calories, x='cluster', y='Calories', labels={'cluster': 'Cluster', 'Calories': 'Average Calories'}, title='Average Calories per Cluster')
    st.plotly_chart(fig)

    def generate_pdf(schedule_data, bmi, weight_category, name, age):
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=letter)
        
        # Set font styles and sizes
        title_font = "Helvetica-Bold"
        title_size = 16
        subtitle_font = "Helvetica-Bold"
        subtitle_size = 12
        normal_font = "Helvetica"
        normal_size = 12
        
        # Set colors
        title_color = colors.blue
        line_color = colors.black
        
        # Title
        c.setFont(title_font, title_size)
        c.setFillColor(title_color)
        c.drawString(100, 750, "DIET SCHEDULE")
        c.line(100, 745, 500, 745)  # Draw a horizontal line under the title
        
        # Name and Age
        c.setFont(subtitle_font, subtitle_size)
        c.setFillColor(line_color)
        c.drawString(100, 730, f"Name: {name}, Age: {age}")
        c.line(100, 725, 500, 725)  # Draw a horizontal line under the name and age
        
        # BMI and Weight Category
        c.setFont(subtitle_font, subtitle_size)
        c.drawString(100, 710, f"BMI: {bmi:.2f} - Weight Category: {weight_category}")
        c.line(100, 705, 500, 705)  # Draw a horizontal line under the BMI and weight category
        
        # Generated by
        c.setFont(subtitle_font, subtitle_size)
        c.drawString(100, 690, "Generated by: WELLNESS-WAYFINDER")
        c.line(100, 685, 500, 685)  # Draw a horizontal line under the generated by
        
        # Vertical space
        c.drawString(100, 670, "")  # Add vertical space
        
        # Diet Schedule Table
        y_start = 650
        for day in schedule_data.index:
            c.setFont(normal_font, normal_size)
            c.drawString(100, y_start, day)
            c.drawString(200, y_start, schedule_data.loc[day, 'Breakfast'])
            c.drawString(350, y_start, schedule_data.loc[day, 'Lunch'])
            c.drawString(500, y_start, schedule_data.loc[day, 'Dinner'])
            y_start -= 20
        
        c.save()
        buffer.seek(0)
        return buffer
    
    # Generate PDF and provide download button
    pdf_buffer = generate_pdf(schedule, bmi, weight_category, name, age)
    st.download_button(label="Download PDF", data=pdf_buffer, file_name="diet_schedule.pdf", mime="application/pdf")
