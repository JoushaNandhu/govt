import streamlit as st
import os
import json
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from pathlib import Path
import hashlib

# Initialize session state variables if they don't exist
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'study_progress' not in st.session_state:
    st.session_state.study_progress = {}
if 'test_results' not in st.session_state:
    st.session_state.test_results = {}

# Create necessary directories if they don't exist
def create_directories():
    directories = ['users', 'study_materials', 'tests', 'tracking']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

# Function to save user data
def save_user_data():
    user_file = Path(f"users/{st.session_state.current_user}.json")
    with open(user_file, 'w') as f:
        json.dump({
            'progress': st.session_state.study_progress.get(st.session_state.current_user, {}),
            'test_results': st.session_state.test_results.get(st.session_state.current_user, [])
        }, f)

# Function to load user data
def load_user_data(username):
    user_file = Path(f"users/{username}.json")
    if user_file.exists():
        with open(user_file, 'r') as f:
            data = json.load(f)
            st.session_state.study_progress[username] = data.get('progress', {})
            st.session_state.test_results[username] = data.get('test_results', [])
    else:
        st.session_state.study_progress[username] = {}
        st.session_state.test_results[username] = []

# Initialize API clients
def init_ai_clients():
    openai_client = None
    perplexity_client = None
    
    if 'OPENAI_API_KEY' in os.environ:
        openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
    
    if 'PPLX_API_KEY' in os.environ:
        # For Perplexity API, we can use the OpenAI client with a different base URL
        perplexity_client = OpenAI(
            api_key=os.environ['PPLX_API_KEY'],
            base_url="https://api.perplexity.ai"
        )
    
    return openai_client, perplexity_client

# Function to generate study material using AI
def generate_study_material(topic, subtopic, client_type="openai"):
    client, perplexity_client = init_ai_clients()
    
    prompt = f"""Generate comprehensive study material for government exam preparation.
    Topic: {topic}
    Subtopic: {subtopic}
    
    Include:
    1. Key concepts and definitions
    2. Important facts and figures
    3. Historical context (if applicable)
    4. Common questions asked in exams
    5. Mnemonics or tips to remember the content
    
    Format the response with proper headers, bullet points, and sections for easy reading.
    """
    
    try:
        if client_type == "openai" and client:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an expert tutor for government exam preparation."},
                          {"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content
        elif client_type == "perplexity" and perplexity_client:
            response = perplexity_client.chat.completions.create(
                model="llama-3-sonar-small-32k-online",
                messages=[{"role": "system", "content": "You are an expert tutor for government exam preparation."},
                          {"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content
        else:
            return "API client not available. Please check your API keys."
    except Exception as e:
        return f"Error generating study material: {str(e)}"

# Function to generate practice questions
def generate_practice_questions(topic, subtopic, num_questions=5, client_type="openai"):
    client, perplexity_client = init_ai_clients()
    
    prompt = f"""Generate {num_questions} multiple-choice practice questions for government exam preparation.
    Topic: {topic}
    Subtopic: {subtopic}
    
    For each question, provide:
    1. The question
    2. Four options (A, B, C, D)
    3. The correct answer
    4. A brief explanation of why the answer is correct
    
    Format each question clearly and number them sequentially.
    """
    
    try:
        if client_type == "openai" and client:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": "You are an expert question generator for government exams."},
                          {"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content
        elif client_type == "perplexity" and perplexity_client:
            response = perplexity_client.chat.completions.create(
                model="llama-3-sonar-small-32k-online",
                messages=[{"role": "system", "content": "You are an expert question generator for government exams."},
                          {"role": "user", "content": prompt}],
                max_tokens=2000
            )
            return response.choices[0].message.content
        else:
            return "API client not available. Please check your API keys."
    except Exception as e:
        return f"Error generating practice questions: {str(e)}"

# Function to parse practice questions and present them as a quiz
def present_quiz(questions_text):
    st.markdown("## Practice Quiz")
    
    lines = questions_text.split('\n')
    questions = []
    current_question = {}
    options = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check if line starts with a number followed by a period (new question)
        if line[0].isdigit() and '.' in line[:3]:
            if current_question and 'question' in current_question:
                current_question['options'] = options
                questions.append(current_question)
                current_question = {}
                options = []
            current_question['question'] = line
        
        # Check for options
        elif line.startswith(('A)', 'B)', 'C)', 'D)', 'A.', 'B.', 'C.', 'D.')):
            options.append(line)
            
        # Check for correct answer indicator
        elif line.lower().startswith('correct answer') or line.lower().startswith('answer:'):
            correct_option = line.split(':')[-1].strip()
            if correct_option in ['A', 'B', 'C', 'D']:
                current_question['correct'] = correct_option
            
        # Explanation
        elif line.lower().startswith('explanation'):
            current_question['explanation'] = line
    
    # Add the last question
    if current_question and 'question' in current_question:
        current_question['options'] = options
        questions.append(current_question)
    
    # Present the quiz
    user_answers = {}
    for i, q in enumerate(questions):
        st.markdown(f"### {q['question']}")
        for opt in q.get('options', []):
            st.markdown(f"{opt}")
        
        answer = st.radio("Your answer:", ["A", "B", "C", "D"], key=f"q_{i}")
        user_answers[i] = answer
        
        if st.button("Check Answer", key=f"check_{i}"):
            if answer == q.get('correct', ''):
                st.success("Correct!")
            else:
                st.error(f"Incorrect. The correct answer is {q.get('correct', '')}.")
            
            if 'explanation' in q:
                st.info(q['explanation'])
    
    if st.button("Submit Quiz"):
        score = sum(1 for i, ans in user_answers.items() if ans == questions[i].get('correct', ''))
        st.success(f"Your score: {score}/{len(questions)}")
        
        # Save test result
        if st.session_state.current_user:
            st.session_state.test_results.setdefault(st.session_state.current_user, []).append({
                'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                'topic': "Practice Quiz",
                'score': score,
                'total': len(questions)
            })
            save_user_data()

# Function to track user study progress
def track_study_time(topic, duration_minutes):
    if st.session_state.current_user:
        today = datetime.datetime.now().strftime("%Y-%m-%d")
        
        # Initialize nested dictionaries if they don't exist
        if st.session_state.current_user not in st.session_state.study_progress:
            st.session_state.study_progress[st.session_state.current_user] = {}
        
        if today not in st.session_state.study_progress[st.session_state.current_user]:
            st.session_state.study_progress[st.session_state.current_user][today] = {}
            
        # Add study time
        if topic in st.session_state.study_progress[st.session_state.current_user][today]:
            st.session_state.study_progress[st.session_state.current_user][today][topic] += duration_minutes
        else:
            st.session_state.study_progress[st.session_state.current_user][today][topic] = duration_minutes
            
        save_user_data()
        st.success(f"Recorded {duration_minutes} minutes of study time for {topic}!")

# Function to display study progress
def display_progress():
    if not st.session_state.current_user or st.session_state.current_user not in st.session_state.study_progress:
        st.warning("No study data available yet.")
        return
        
    progress_data = st.session_state.study_progress[st.session_state.current_user]
    
    # Convert to DataFrame for easier manipulation
    rows = []
    for date, topics in progress_data.items():
        for topic, minutes in topics.items():
            rows.append({"Date": date, "Topic": topic, "Minutes": minutes})
    
    if not rows:
        st.warning("No study data available yet.")
        return
        
    df = pd.DataFrame(rows)
    
    # Display total study time by topic
    st.subheader("Total Study Time by Topic")
    topic_totals = df.groupby("Topic")["Minutes"].sum().reset_index()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(topic_totals["Topic"], topic_totals["Minutes"])
    ax.set_ylabel("Minutes")
    ax.set_title("Total Study Time by Topic")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
    
    # Display daily study time
    st.subheader("Daily Study Progress")
    daily_totals = df.groupby("Date")["Minutes"].sum().reset_index()
    daily_totals = daily_totals.sort_values("Date")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(daily_totals["Date"], daily_totals["Minutes"], marker='o')
    ax2.set_ylabel("Minutes")
    ax2.set_title("Daily Study Time")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig2)
    
    # Display recent activity as a table
    st.subheader("Recent Study Activity")
    df["Date"] = pd.to_datetime(df["Date"])
    recent_df = df.sort_values("Date", ascending=False).head(10)
    st.table(recent_df[["Date", "Topic", "Minutes"]])

# Function to display test results
def display_test_results():
    if not st.session_state.current_user or st.session_state.current_user not in st.session_state.test_results:
        st.warning("No test results available yet.")
        return
        
    results = st.session_state.test_results[st.session_state.current_user]
    
    if not results:
        st.warning("No test results available yet.")
        return
        
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Display results as a table
    st.subheader("Your Test Results")
    df["Percentage"] = (df["score"] / df["total"]) * 100
    df["Percentage"] = df["Percentage"].round(2)
    st.table(df[["date", "topic", "score", "total", "Percentage"]])
    
    # Display performance over time
    st.subheader("Test Performance Over Time")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df["date"], df["Percentage"], marker='o')
    ax.set_ylabel("Score (%)")
    ax.set_title("Test Performance Trend")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)

# Authentication functions
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def authenticate(username, password):
    user_file = Path(f"users/{username}.json")
    if not user_file.exists():
        return False
    
    with open(user_file, 'r') as f:
        data = json.load(f)
        return data.get('password_hash') == hash_password(password)

def create_user(username, password):
    user_file = Path(f"users/{username}.json")
    if user_file.exists():
        return False
    
    with open(user_file, 'w') as f:
        json.dump({
            'username': username,
            'password_hash': hash_password(password),
            'progress': {},
            'test_results': []
        }, f)
    return True

# Login UI
def login_ui():
    st.title("Government Exam Preparation Platform")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.header("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.current_user = username
                load_user_data(username)
                st.success("Login successful!")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password.")
                
    with tab2:
        st.header("Register")
        new_username = st.text_input("Choose Username", key="reg_username")
        new_password = st.text_input("Choose Password", type="password", key="reg_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Register"):
            if new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                if create_user(new_username, new_password):
                    st.success("Registration successful! You can now login.")
                else:
                    st.error("Username already exists. Please choose another one.")

# Main application UI
def main_app():
    st.title(f"Welcome to Government Exam Prep, {st.session_state.current_user}!")
    
    # Sidebar menu
    menu = st.sidebar.selectbox(
        "Navigation",
        ["Study Materials", "Practice Tests", "Track Progress", "Settings", "Logout"]
    )
    
    if menu == "Study Materials":
        st.header("Study Materials")
        
        # Subject selection
        subject_options = [
            "General Knowledge", "Current Affairs", "History", "Geography", 
            "Indian Polity", "Economy", "Science & Technology", "Environment & Ecology"
        ]
        
        selected_subject = st.selectbox("Select Subject", subject_options)
        
        # Subtopic selection based on the selected subject
        subtopics = {
            "General Knowledge": ["Basic Facts", "Important Dates", "Awards & Honors", "Books & Authors"],
            "Current Affairs": ["National", "International", "Sports", "Business"],
            "History": ["Ancient India", "Medieval India", "Modern India", "World History"],
            "Geography": ["Physical Geography", "Indian Geography", "World Geography", "Economic Geography"],
            "Indian Polity": ["Constitution", "Parliament", "Judiciary", "Local Government"],
            "Economy": ["Basics of Economy", "Indian Economy", "Banking", "International Organizations"],
            "Science & Technology": ["Physics", "Chemistry", "Biology", "Technology"],
            "Environment & Ecology": ["Biodiversity", "Climate Change", "Conservation", "Pollution"]
        }
        
        selected_subtopic = st.selectbox("Select Topic", subtopics.get(selected_subject, []))
        
        # AI provider selection
        ai_provider = st.radio("Select AI Provider", ["OpenAI", "Perplexity"])
        client_type = "openai" if ai_provider == "OpenAI" else "perplexity"
        
        if st.button("Generate Study Material"):
            with st.spinner("Generating study material..."):
                material = generate_study_material(selected_subject, selected_subtopic, client_type)
                st.markdown(material)
                
                # Option to record study time
                st.subheader("Record Your Study Time")
                study_time = st.number_input("Study duration (minutes)", min_value=5, max_value=240, value=30, step=5)
                if st.button("Record Study Time"):
                    track_study_time(f"{selected_subject} - {selected_subtopic}", study_time)
    
    elif menu == "Practice Tests":
        st.header("Practice Tests")
        
        test_options = [
            "Quick Practice Quiz", 
            "Subject-specific Test", 
            "Mock Exam"
        ]
        
        test_type = st.selectbox("Select Test Type", test_options)
        
        if test_type == "Quick Practice Quiz":
            subject_options = [
                "General Knowledge", "Current Affairs", "History", "Geography", 
                "Indian Polity", "Economy", "Science & Technology", "Environment & Ecology"
            ]
            
            selected_subject = st.selectbox("Select Subject", subject_options)
            
            # Subtopic selection based on the selected subject
            subtopics = {
                "General Knowledge": ["Basic Facts", "Important Dates", "Awards & Honors", "Books & Authors"],
                "Current Affairs": ["National", "International", "Sports", "Business"],
                "History": ["Ancient India", "Medieval India", "Modern India", "World History"],
                "Geography": ["Physical Geography", "Indian Geography", "World Geography", "Economic Geography"],
                "Indian Polity": ["Constitution", "Parliament", "Judiciary", "Local Government"],
                "Economy": ["Basics of Economy", "Indian Economy", "Banking", "International Organizations"],
                "Science & Technology": ["Physics", "Chemistry", "Biology", "Technology"],
                "Environment & Ecology": ["Biodiversity", "Climate Change", "Conservation", "Pollution"]
            }
            
            selected_subtopic = st.selectbox("Select Topic", subtopics.get(selected_subject, []))
            num_questions = st.slider("Number of Questions", 5, 20, 10)
            
            # AI provider selection
            ai_provider = st.radio("Select AI Provider", ["OpenAI", "Perplexity"])
            client_type = "openai" if ai_provider == "OpenAI" else "perplexity"
            
            if st.button("Generate Quiz"):
                with st.spinner("Generating practice questions..."):
                    questions = generate_practice_questions(selected_subject, selected_subtopic, num_questions, client_type)
                    present_quiz(questions)
                    
        elif test_type == "Subject-specific Test":
            st.info("Coming soon! This feature will allow comprehensive tests on specific subjects.")
            
        elif test_type == "Mock Exam":
            st.info("Coming soon! This feature will simulate a full government exam experience.")
    
    elif menu == "Track Progress":
        st.header("Your Study Progress")
        
        tab1, tab2 = st.tabs(["Study Time", "Test Results"])
        
        with tab1:
            display_progress()
            
        with tab2:
            display_test_results()
    
    elif menu == "Settings":
        st.header("Settings")
        
        # API Key settings
        st.subheader("API Keys")
        openai_key = st.text_input("OpenAI API Key", type="password", value=os.environ.get("OPENAI_API_KEY", ""))
        pplx_key = st.text_input("Perplexity API Key", type="password", value=os.environ.get("PPLX_API_KEY", ""))
        
        if st.button("Save API Keys"):
            os.environ["OPENAI_API_KEY"] = openai_key
            os.environ["PPLX_API_KEY"] = pplx_key
            st.success("API keys saved successfully!")
        
        # User profile settings
        st.subheader("User Profile")
        st.info("More user profile settings coming soon!")
    
    elif menu == "Logout":
        st.session_state.current_user = None
        st.success("Logged out successfully!")
        st.experimental_rerun()

# Main function
def main():
    create_directories()
    
    # Check if user is logged in
    if st.session_state.current_user:
        main_app()
    else:
        login_ui()

if __name__ == "__main__":
    main()
