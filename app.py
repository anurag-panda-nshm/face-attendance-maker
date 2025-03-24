import streamlit as st
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import datetime
import pickle
from PIL import Image
import time

# Set page configuration
st.set_page_config(
    page_title="Facial Recognition Attendance System",
    page_icon="📊",
    layout="wide"
)

# Create necessary directories if they don't exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("data/faces"):
    os.makedirs("data/faces")
if not os.path.exists("data/attendance"):
    os.makedirs("data/attendance")

# Load or create student database
def load_student_data():
    if os.path.exists("data/students.pkl"):
        with open("data/students.pkl", "rb") as f:
            return pickle.load(f)
    else:
        return pd.DataFrame(columns=["student_id", "name", "department", "year", "face_encoding"])

def save_student_data(df):
    with open("data/students.pkl", "wb") as f:
        pickle.dump(df, f)

# Load attendance data for a specific date
def load_attendance_data(date_str):
    file_path = f"data/attendance/{date_str}.csv"
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        # Create a new attendance dataframe with student data
        students_df = load_student_data()
        if len(students_df) > 0:
            attendance_df = students_df[["student_id", "name", "department", "year"]].copy()
            attendance_df["status"] = "Absent"
            attendance_df["time"] = ""
            return attendance_df
        return pd.DataFrame(columns=["student_id", "name", "department", "year", "status", "time"])

def save_attendance_data(df, date_str):
    file_path = f"data/attendance/{date_str}.csv"
    df.to_csv(file_path, index=False)

# Navigation
def main():
    st.sidebar.title("Navigation")
    pages = ["Student Registration", "Attendance", "Reports"]
    choice = st.sidebar.selectbox("Go to", pages)

    if choice == "Student Registration":
        student_registration_page()
    elif choice == "Attendance":
        attendance_page()
    else:
        reports_page()

def student_registration_page():
    st.title("Student Registration")
    
    # Load existing student data
    students_df = load_student_data()
    
    # Display existing students in a table
    if not students_df.empty:
        st.subheader("Registered Students")
        display_df = students_df.drop(columns=["face_encoding"]) if "face_encoding" in students_df.columns else students_df
        st.dataframe(display_df)
    
    # Form for adding new students
    st.subheader("Add New Student")
    
    col1, col2 = st.columns(2)
    
    with col1:
        student_id = st.text_input("Student ID")
        name = st.text_input("Full Name")
        department = st.text_input("Department")
        year = st.selectbox("Year", [1, 2, 3, 4])
    
    with col2:
        st.write("Capture Face Image")
        img_file_buffer = st.camera_input("Take a picture")
        
        if img_file_buffer is not None:
            # Convert the file buffer to an image
            img = Image.open(img_file_buffer)
            img_array = np.array(img)
            
            # Display the captured image
            st.image(img_array, caption="Captured Image", use_column_width=True)
            
            # Detect faces in the image
            face_locations = face_recognition.face_locations(img_array)
            
            if len(face_locations) == 0:
                st.error("No face detected. Please try again.")
            elif len(face_locations) > 1:
                st.error("Multiple faces detected. Please ensure only one face is in the frame.")
            else:
                st.success("Face detected successfully!")
                face_encoding = face_recognition.face_encodings(img_array, face_locations)[0]
                
                if st.button("Register Student"):
                    if student_id and name and department:
                        # Check if student ID already exists
                        if student_id in students_df["student_id"].values:
                            st.error(f"Student ID {student_id} already exists.")
                        else:
                            # Save student data
                            new_student = {
                                "student_id": student_id,
                                "name": name,
                                "department": department,
                                "year": year,
                                "face_encoding": face_encoding
                            }
                            students_df = pd.concat([students_df, pd.DataFrame([new_student])], ignore_index=True)
                            save_student_data(students_df)
                            
                            # Save face image
                            img.save(f"data/faces/{student_id}.jpg")
                            
                            st.success(f"Student {name} registered successfully!")
                            st.experimental_rerun()
                    else:
                        st.warning("Please fill all required fields: Student ID, Name, and Department.")

def attendance_page():
    st.title("Take Attendance")
    
    # Load student data
    students_df = load_student_data()
    
    if students_df.empty:
        st.warning("No students registered. Please register students first.")
        return
    
    # Select date for attendance
    today = datetime.date.today()
    date = st.date_input("Select Date", today)
    date_str = date.strftime("%Y-%m-%d")
    
    # Load attendance data for selected date
    attendance_df = load_attendance_data(date_str)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Capture Attendance")
        start_camera = st.button("Start Camera")
        stop_camera = st.button("Stop Camera")
        
        placeholder = st.empty()
        
        if start_camera:
            # Initialize webcam
            cap = cv2.VideoCapture(0)
            
            # Get face encodings from all registered students
            known_face_encodings = np.array(students_df["face_encoding"].tolist())
            known_student_ids = students_df["student_id"].tolist()
            
            # Set to keep track of recognized students to avoid duplicates
            recognized_students = set()
            
            while not stop_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture image from camera.")
                    break
                
                # Convert the image from BGR to RGB for face_recognition
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Find faces in the current frame
                face_locations = face_recognition.face_locations(rgb_frame)
                
                if face_locations:
                    # Get face encodings for faces in the current frame
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    
                    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
                        # Compare with known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                        
                        if True in matches:
                            # Get the index of the first matching face
                            match_index = matches.index(True)
                            student_id = known_student_ids[match_index]
                            
                            # Get student info
                            student_info = students_df[students_df["student_id"] == student_id].iloc[0]
                            student_name = student_info["name"]
                            
                            # Draw rectangle and text on the frame
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            cv2.putText(frame, student_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            
                            # Mark attendance if not already marked
                            if student_id not in recognized_students:
                                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                                attendance_df.loc[attendance_df["student_id"] == student_id, "status"] = "Present"
                                attendance_df.loc[attendance_df["student_id"] == student_id, "time"] = current_time
                                recognized_students.add(student_id)
                                
                                # Save attendance data
                                save_attendance_data(attendance_df, date_str)
                                
                                # Show notification
                                st.sidebar.success(f"Attendance marked for {student_name}")
                        else:
                            # Unrecognized face
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Display the frame
                placeholder.image(frame, channels="BGR", use_column_width=True)
                
                time.sleep(0.1)  # Short delay to reduce CPU usage
            
            cap.release()
    
    with col2:
        st.subheader("Attendance Register")
        
        # Display attendance data with a filter
        filter_option = st.selectbox("Filter by Status", ["All", "Present", "Absent"])
        
        filtered_df = attendance_df
        if filter_option != "All":
            filtered_df = attendance_df[attendance_df["status"] == filter_option]
        
        st.dataframe(filtered_df)
        
        # Manual attendance marking
        st.subheader("Mark Attendance Manually")
        
        # Create a multiselect for absent students
        absent_students = attendance_df[attendance_df["status"] == "Absent"]
        if not absent_students.empty:
            selected_students = st.multiselect(
                "Select students to mark as present",
                absent_students["student_id"].tolist(),
                format_func=lambda x: f"{x} - {absent_students[absent_students['student_id'] == x]['name'].values[0]}"
            )
            
            if selected_students and st.button("Mark Selected as Present"):
                current_time = datetime.datetime.now().strftime("%H:%M:%S")
                for student_id in selected_students:
                    attendance_df.loc[attendance_df["student_id"] == student_id, "status"] = "Present"
                    attendance_df.loc[attendance_df["student_id"] == student_id, "time"] = current_time
                
                save_attendance_data(attendance_df, date_str)
                st.success(f"Marked {len(selected_students)} students as present.")
                st.experimental_rerun()

def reports_page():
    st.title("Attendance Reports")
    
    # Load student data
    students_df = load_student_data()
    
    if students_df.empty:
        st.warning("No students registered. Please register students first.")
        return
    
    # Get list of all attendance files
    attendance_files = os.listdir("data/attendance")
    attendance_dates = [file.split(".")[0] for file in attendance_files if file.endswith(".csv")]
    attendance_dates.sort(reverse=True)
    
    if not attendance_dates:
        st.warning("No attendance records found.")
        return
    
    # Date selection
    selected_date = st.selectbox("Select Date", attendance_dates)
    
    # Load attendance data for selected date
    attendance_df = load_attendance_data(selected_date)
    
    # Display statistics
    st.subheader("Attendance Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    total_students = len(attendance_df)
    present_students = len(attendance_df[attendance_df["status"] == "Present"])
    absent_students = total_students - present_students
    
    with col1:
        st.metric("Total Students", total_students)
    
    with col2:
        st.metric("Present", present_students)
    
    with col3:
        st.metric("Absent", absent_students)
    
    # Display attendance data
    st.subheader("Attendance Register")
    st.dataframe(attendance_df)
    
    # Download attendance report
    csv = attendance_df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name=f"attendance_{selected_date}.csv",
        mime="text/csv"
    )
    
    # Department-wise attendance
    st.subheader("Department-wise Attendance")
    dept_stats = attendance_df.groupby("department").agg(
        Total=("student_id", "count"),
        Present=("status", lambda x: (x == "Present").sum()),
        Absent=("status", lambda x: (x == "Absent").sum())
    )
    dept_stats["Percentage"] = (dept_stats["Present"] / dept_stats["Total"] * 100).round(2)
    
    st.dataframe(dept_stats)
    
    # Year-wise attendance
    st.subheader("Year-wise Attendance")
    year_stats = attendance_df.groupby("year").agg(
        Total=("student_id", "count"),
        Present=("status", lambda x: (x == "Present").sum()),
        Absent=("status", lambda x: (x == "Absent").sum())
    )
    year_stats["Percentage"] = (year_stats["Present"] / year_stats["Total"] * 100).round(2)
    
    st.dataframe(year_stats)

if __name__ == "__main__":
    main()