import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import random

# Load your pre-trained model
model = load_model('gymmodel.keras')
data_cat = ['barbell', 'dumbbell', 'gym_ball', 'kettlebell', 'smith_machine']  # Update with your actual classes

# Exercise data (update this with actual data)
machine_exercises = {
    'barbell': {
        'upper_body': ['Bench Press', 'Barbell Rows', 'Overhead Press'],
        'lower_body': ['Squats', 'Deadlifts', 'Lunges'],
        'core': ['Russian Twists', 'Barbell Rollouts', 'Hanging Leg Raises']
    },
    'dumbbell': {
        'upper_body': ['Dumbbell Chest Press', 'Shoulder Press', 'Dumbbell Rows'],
        'lower_body': ['Dumbbell Squats', 'Dumbbell Romanian Deadlifts', 'Step-ups'],
        'core': ['Dumbbell Side Bends', 'Dumbbell Woodchoppers', 'Plank Rows']
    },
    'gym_ball': {
        'core': ['Ball Crunches', 'Plank with Leg Lifts', 'Russian Twists']
    },
    'kettlebell': {
        'upper_body': ['Kettlebell Swings', 'Kettlebell Press', 'Kettlebell Rows'],
        'lower_body': ['Kettlebell Goblet Squats', 'Kettlebell Lunges', 'Kettlebell Deadlifts'],
        'core': ['Kettlebell Russian Twists', 'Kettlebell Windmills', 'Kettlebell Turkish Get-ups']
    },
    'smith_machine': {
        'upper_body': ['Smith Machine Bench Press', 'Smith Machine Shoulder Press', 'Smith Machine Rows'],
        'lower_body': ['Smith Machine Squats', 'Smith Machine Lunges', 'Smith Machine Deadlifts']
    }
}

exercise_links = {
    'Bench Press': 'https://www.youtube.com/shorts/0cXAp6WhSj4',
    'Barbell Rows': 'https://www.youtube.com/shorts/Nqh7q3zDCoQ',
    'Overhead Press': 'https://www.youtube.com/shorts/zSU7T1zZagQ',
    'Squats': 'https://www.youtube.com/shorts/gslEzVggur8',
    'Deadlifts': 'https://www.youtube.com/shorts/8np3vKDBJfc',
    'Lunges': 'https://www.youtube.com/shorts/TwEH620Pn6A',
    'Russian Twists': 'https://www.youtube.com/watch?v=Tau0hsW8iR0',
    'Barbell Rollouts': 'https://www.youtube.com/watch?v=3C1TRMJveXo',
    'Hanging Leg Raises': 'https://www.youtube.com/shorts/2n4UqRIJyk4',
    'Dumbbell Chest Press': 'https://www.youtube.com/shorts/SidmT09GXz8',
    'Shoulder Press': 'https://www.youtube.com/shorts/dyv6g4xBFGU',
    'Dumbbell Rows': 'https://www.youtube.com/shorts/s1H87k4tAaA',
    'Dumbbell Squats': 'https://www.youtube.com/shorts/eLX_dyvooKQ',
    'Dumbbell Romanian Deadlifts': 'https://www.youtube.com/shorts/u14AwrUcwWw',
    'Step-ups': 'https://www.youtube.com/shorts/PzDbmqL6qo8Dumbbell',
    'Dumbbell Side Bends': 'https://www.youtube.com/watch?v=dL9ZzqtQI5c',
    'Dumbbell Woodchoppers': 'https://www.youtube.com/shorts/OgQU_bbdB7c',
    'Plank Rows': 'https://www.youtube.com/watch?v=Gtc_Ns3qYYo',
    'Ball Crunches': 'https://www.youtube.com/watch?v=O4d3kd1ZLyc',
    'Plank with Leg Lifts': 'https://www.youtube.com/shorts/s_8PheAKUYk',
    'Kettlebell Swings': 'https://www.youtube.com/shorts/SR_4kUbkEaw',
    'Kettlebell Press': 'https://www.youtube.com/watch?v=eKQ0JOx_1qI',
    'Kettlebell Rows': 'https://www.youtube.com/shorts/e4OSLk1qZOQ',
    'Kettlebell Goblet Squats': 'https://www.youtube.com/shorts/dBnNCOtuGNQ',
    'Kettlebell Lunges': 'https://www.youtube.com/shorts/otd2YQk7osI',
    'Kettlebell Deadlifts': 'https://www.youtube.com/shorts/I7q_EPywprs',
    'Kettlebell Russian Twists': 'https://www.youtube.com/shorts/BA-uP_-bVE8',
    'Kettlebell Windmills': 'https://www.youtube.com/shorts/OVNXkKsfy7o',
    'Kettlebell Turkish Get-ups': 'https://www.youtube.com/shorts/-dfk79o6iHI',
    'Smith Machine Bench Press': 'https://www.youtube.com/shorts/G-jT0m0nvx8',
    'Smith Machine Shoulder Press': 'https://www.youtube.com/shorts/QWdaC7rQ-FM',
    'Smith Machine Rows': 'https://www.youtube.com/shorts/qivPkcDI0s0',
    'Smith Machine Squats': 'https://www.youtube.com/shorts/xU4cuTffVZc',
    'Smith Machine Lunges': 'https://www.youtube.com/shorts/dFMa-mmZ6A8',
    'Smith Machine Deadlifts': 'https://www.youtube.com/shorts/Bhg9IvQzsCI'
}


exercise_descriptions = {
    'Bench Press': 'A great exercise for building chest and tricep strength.',
    'Barbell Rows': 'Works your back muscles and improves posture.',
    'Overhead Press': 'Targets shoulders and triceps, promoting upper body strength.',
    'Squats': 'Essential for building lower body strength and muscle mass.',
    'Deadlifts': 'Targets multiple muscle groups, especially the back and legs.',
    'Lunges': 'Excellent for building leg strength and balance.',
    'Russian Twists': 'Great for working the oblique muscles.',
    'Barbell Rollouts': 'An advanced exercise for core strength.',
    'Hanging Leg Raises': 'Targets lower abs and hip flexors.',
    'Dumbbell Chest Press': 'Helps build chest and tricep muscles.',
    'Shoulder Press': 'Enhances shoulder strength and stability.',
    'Dumbbell Rows': 'Strengthens back muscles and improves posture.',
    'Dumbbell Squats': 'Builds leg muscles and improves balance.',
    'Dumbbell Romanian Deadlifts': 'Focuses on hamstrings and glutes.',
    'Step-ups': 'Great for leg strength and cardiovascular health.',
    'Dumbbell Side Bends': 'Targets the oblique muscles for better core strength.',
    'Dumbbell Woodchoppers': 'Works the core and improves rotational strength.',
    'Plank Rows': 'Combines core and upper body strength training.',
    'Ball Crunches': 'Targets the abs for core strength.',
    'Plank with Leg Lifts': 'Combines core stability and leg strength.',
    'Kettlebell Swings': 'A full-body exercise that improves strength and conditioning.',
    'Kettlebell Press': 'Builds shoulder and tricep strength.',
    'Kettlebell Rows': 'Targets back muscles for improved posture.',
    'Kettlebell Goblet Squats': 'Focuses on leg strength and core stability.',
    'Kettlebell Lunges': 'Enhances leg strength and balance.',
    'Kettlebell Deadlifts': 'Strengthens the back and legs.',
    'Kettlebell Russian Twists': 'Works the oblique muscles.',
    'Kettlebell Windmills': 'Improves core strength and shoulder stability.',
    'Kettlebell Turkish Get-ups': 'A full-body exercise that improves strength and coordination.',
    'Smith Machine Bench Press': 'Targets the chest and triceps with added stability.',
    'Smith Machine Shoulder Press': 'Enhances shoulder strength with controlled movement.',
    'Smith Machine Rows': 'Strengthens back muscles with stability.',
    'Smith Machine Squats': 'Focuses on leg strength with controlled movement.',
    'Smith Machine Lunges': 'Builds leg strength and balance.',
    'Smith Machine Deadlifts': 'Targets back and leg muscles with added stability.'
}

recommendations_pool = {
    'upper_body': [
        "Try incorporating more push-up variations for added upper body strength.",
        "Consider adding pull-ups to enhance your back and bicep muscles.",
        "Include shoulder exercises like lateral raises for well-rounded upper body development."
    ],
    'lower_body': [
        "Include calf raises to improve lower leg strength and endurance.",
        "Add lunges and step-ups to your routine to target different leg muscles.",
        "Incorporate plyometric exercises like box jumps for explosive leg power."
    ],
    'core': [
        "Mix in planks and side planks to enhance core stability.",
        "Try ab rollout exercises for a challenging core workout.",
        "Include leg raises and hanging exercises to target lower abs."
    ],
    'general': [
        "Balance your routine with cardio exercises like running or cycling.",
        "Incorporate flexibility exercises like yoga or stretching to improve mobility.",
        "Ensure you have rest days to allow muscles to recover and grow."
    ]
}

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(180, 180))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_batch)
    score = tf.nn.softmax(predictions[0])
    return data_cat[np.argmax(score)], np.max(score)

def suggest_exercises(equipment, muscle_group=None):
    if equipment not in machine_exercises:
        return "This equipment is not available for suggestions."

    response = f"Suggested exercises for {equipment}:\n"
    for part, exercises in machine_exercises[equipment].items():
        if muscle_group and muscle_group.lower() != part:
            continue
        response += f"\n{part.capitalize()} exercises:\n"
        for exercise in exercises:
            description = exercise_descriptions.get(exercise, "No description available")
            link = exercise_links.get(exercise, "No link available")
            response += f"- **{exercise}**: {description}\n  [Watch here]({link})\n"
    return response

def rate_exercise(exercise):
    rating = st.slider(f"Rate {exercise}", 1, 5, 3)
    st.write(f"You rated {exercise} a {rating}/5")

def main_page():
    st.title("Exercise Suggestion App")
    uploaded_file = st.file_uploader("Choose an image of exercise equipment", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        with open("dataset/user_upload/uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully.")
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        equipment, confidence = predict_image("dataset/user_upload/uploaded_image.jpg")
        st.write(f"Predicted Equipment: {equipment} with a confidence of {confidence * 100:.2f}%")

        muscle_group = st.selectbox("Select muscle group", ["All", "Upper Body", "Lower Body", "Core"])
        muscle_group = muscle_group.lower().replace(" ", "_") if muscle_group != "All" else None

        suggestions = suggest_exercises(equipment, muscle_group)
        st.markdown(suggestions)

        for part, exercises in machine_exercises.get(equipment, {}).items():
            if muscle_group and muscle_group != part:
                continue
            for exercise in exercises:
                rate_exercise(exercise)

def exercise_page():
    st.title("Exercise Routine")
    selected_exercises = st.multiselect("Select exercises", list(exercise_descriptions.keys()))

    if 'exercise_data' not in st.session_state:
        st.session_state['exercise_data'] = {}

    for exercise in selected_exercises:
        if exercise not in st.session_state['exercise_data']:
            st.session_state['exercise_data'][exercise] = {'start': None, 'end': None, 'duration': None}

        exercise_data = st.session_state['exercise_data'][exercise]

        if exercise_data['start'] is None:
            if st.button(f"Start {exercise}"):
                exercise_data['start'] = datetime.now()

        elif exercise_data['end'] is None:
            st.write(f"Started at: {exercise_data['start']}")
            if st.button(f"Finish {exercise}"):
                exercise_data['end'] = datetime.now()
                exercise_data['duration'] = exercise_data['end'] - exercise_data['start']
                st.success(f"{exercise} completed in {exercise_data['duration']}.")

    if st.button("Finish Routine"):
        st.write("Great job! Here's your performance summary:")
        exercise_summary = []

        for exercise, data in st.session_state['exercise_data'].items():
            if data['duration']:
                exercise_summary.append([exercise, data['start'].strftime("%H:%M:%S"), data['end'].strftime("%H:%M:%S"), str(data['duration'])])

        if exercise_summary:
            st.table(exercise_summary)

        # Collect selected muscle groups from exercises
        selected_muscle_groups = set()
        for exercise in selected_exercises:
            for part, exercises in machine_exercises.items():
                if exercise in exercises:
                    selected_muscle_groups.add(part)

        # Generate random recommendations
        recommendations = []
        for muscle_group in selected_muscle_groups:
            if muscle_group in recommendations_pool:
                recommendations.extend(random.sample(recommendations_pool[muscle_group], 1))

        recommendations.extend(random.sample(recommendations_pool['general'], 2))

        st.markdown("**Here are some recommendations for your next workout:**")
        for rec in recommendations:
            st.markdown(f"- **:blue[{rec}]**")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Main Page", "Exercise Page"])

    if page == "Main Page":
        main_page()
    elif page == "Exercise Page":
        exercise_page()

if __name__ == "__main__":
    main()