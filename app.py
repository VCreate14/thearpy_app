import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="TherapEase - Mental Wellness Activity Predictor",
    page_icon="🧘‍♀️",
    layout="wide"
)

# Dark mode toggle in sidebar with better styling
with st.sidebar:
    dark_mode = st.toggle('Dark Mode 🌙')

# Updated Custom CSS with better dark mode handling
st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-color: {('#0E1117' if dark_mode else '#ffffff')};
    }}
    
    [data-testid="stSidebar"] {{
        background-color: {('#1E2530' if dark_mode else '#f0f2f6')};
    }}
    
    .stTitle {{
        color: {('#ffffff' if dark_mode else '#2E4057')} !important;
    }}
    
    h1, h2, h3, h4, h5, h6, p {{
        color: {('#ffffff' if dark_mode else '#2E4057')} !important;
    }}
    
    div[data-baseweb="select"] > div {{
        background-color: {('#1E2530' if dark_mode else '#ffffff')};
        color: {('#ffffff' if dark_mode else '#2E4057')};
    }}
    
    .stMarkdown {{
        color: {('#ffffff' if dark_mode else '#2E4057')};
    }}
    
    .success-message {{
        background-color: {('#1E2530' if dark_mode else '#E8F5E9')};
        color: {('#ffffff' if dark_mode else '#2E4057')};
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }}
    
    .stButton button {{
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 10px 25px;
        font-size: 18px;
    }}
    
    [data-testid="stMarkdownContainer"] {{
        color: {('#ffffff' if dark_mode else '#2E4057')};
    }}
    
    .stSlider {{
        color: {('#ffffff' if dark_mode else '#2E4057')};
    }}
    
    div[data-baseweb="select"] {{
        background-color: {('#1E2530' if dark_mode else '#ffffff')};
    }}
    
    .stSelectbox label {{
        color: {('#ffffff' if dark_mode else '#2E4057')};
    }}
    </style>
""", unsafe_allow_html=True)

# Load model and labels (same as before)
model = joblib.load('therapease_model.joblib')
labels = {
    'Mood': {
        'classes': ['Angry', 'Sad', 'Happy', 'Anxious'],
        'mapping': {'0': 'Angry', '1': 'Sad', '2': 'Happy', '3': 'Anxious'}
    },
    'Energy': {
        'classes': ['Low', 'Medium', 'High'],
        'mapping': {'0': 'Low', '1': 'Medium', '2': 'High'}
    },
    'Stress': {
        'classes': ['Low', 'Medium', 'High'],
        'mapping': {'0': 'Low', '1': 'Medium', '2': 'High'}
    },
    'SuggestedActivity': {
        'classes': ['Journaling', 'CBT Worksheet', 'Deep Breathing', 'Call Friend', 'Walk', 'Music Therapy', 'Gratitude'],
        'mapping': {'0': 'Journaling', '1': 'CBT Worksheet', '2': 'Deep Breathing', '3': 'Call Friend', '4': 'Walk', '5': 'Music Therapy', '6': 'Gratitude'}
    }
}

def main():
    # Header
    st.title('🧘‍♀️ TherapEase Activity Predictor')
    st.markdown("---")
    
    # Introduction
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h3>Welcome to Your Personal Mental Wellness Assistant</h3>
            <p>Let us help you find the perfect activity based on your current state of mind.</p>
        </div>
    """, unsafe_allow_html=True)

    # Create two columns for input fields
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### How are you feeling?")
        mood = st.selectbox('Select your mood', labels['Mood']['classes'], 
                          help="Choose the emotion that best describes your current state")
        energy = st.selectbox('Select your energy level', labels['Energy']['classes'],
                            help="How energetic do you feel right now?")
        stress = st.selectbox('Select your stress level', labels['Stress']['classes'],
                            help="Rate your current stress level")

    with col2:
        st.markdown("### Your Current Status")
        sleep_hours = st.slider('Hours of sleep last night', 0, 12, 6,
                              help="How many hours did you sleep last night?")
        st.markdown(f"<div style='text-align: center;'>Sleep Quality: {'😴' * int(sleep_hours/3)}</div>", 
                   unsafe_allow_html=True)
        
        time_available = st.slider('Minutes available', 5, 60, 15,
                                 help="How much time can you dedicate to the activity?")
        st.markdown(f"<div style='text-align: center;'>Time Available: ⏰ {time_available} minutes</div>", 
                   unsafe_allow_html=True)

    # Center the predict button
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        predict_button = st.button('Get Your Personalized Activity 🎯')

    if predict_button:
        try:
            # Prediction logic (same as before)
            encoder = LabelEncoder()
            
            encoder.fit(labels['Mood']['classes'])
            mood_encoded = encoder.transform([mood])[0]
            
            encoder.fit(labels['Energy']['classes'])
            energy_encoded = encoder.transform([energy])[0]
            
            encoder.fit(labels['Stress']['classes'])
            stress_encoded = encoder.transform([stress])[0]

            input_data = pd.DataFrame([[
                mood_encoded,
                energy_encoded,
                stress_encoded,
                sleep_hours,
                time_available
            ]], columns=['Mood', 'Energy', 'Stress', 'SleepHours', 'TimeAvailable'])

            prediction = model.predict(input_data)[0]
            num_activities = len(labels['SuggestedActivity']['classes'])
            prediction = int(prediction) % num_activities
            suggested_activity = labels['SuggestedActivity']['classes'][prediction]

            # Display result with enhanced styling
            st.markdown("---")
            st.markdown(f"""
                <div class='success-message'>
                    <h2 style='text-align: center; color: #4CAF50;'>Suggested Activity</h2>
                    <h1 style='text-align: center; color: #2E4057;'>{suggested_activity}</h1>
                    <p style='text-align: center;'>Take {time_available} minutes to focus on this activity for better mental wellness.</p>
                </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 20px; color: #666;'>
            <p>Remember: Your mental health matters. Take one step at a time.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()