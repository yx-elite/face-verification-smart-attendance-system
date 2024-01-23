import cv2
import os
import csv
import sys
import keyboard
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.layers import Layer
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input


input_img_dir = 'app_data/temp'
verification_img_dir = 'app_data/verification_images'
attendance_log_dir = 'app_data/attendance_log'

face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')
model_name = 'model/siamesemodel_xhlayer_224_data_augmented_epoch30_95samples.h5'

# Global variables
validation_count = 10
verification_threshold = 0.70
detection_threshold = 0.60


global_threshold = 0


def preprocess_image(image_path):
    byte_img = cv2.imread(image_path)
    img = cv2.cvtColor(byte_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    return img

# Create L1 (Manhattan) Distance class
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

siamese_model = load_model(model_name, custom_objects = {'L1Dist': L1Dist})

def verify_face(face_input, face_verification, frame, username, x, y, w, h):
    global global_threshold

    result = siamese_model.predict([np.expand_dims(face_input, axis=0),
                                                np.expand_dims(face_verification, axis=0)])

    if result[0][0] > detection_threshold:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, str(username), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        print("Verified... they are the same person")
        print(f"Similarity Score: {result[0][0]}")
        print("Verification Successful")
        global_threshold += 1
        print(f"Image passes\t: {global_threshold}/{validation_count}")
    else:
        print("Unverified! They are not the same person!")
        print(f"Similarity Score: {result[0][0]}")
        print("Verification Failed")
        print(f"Image passes\t: {global_threshold}/{validation_count}")


def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture


def record_attendance(username):
    # Generate current date, day, and timestamp
    current_date = datetime.now().strftime("%Y-%m-%d")
    timestamp = datetime.now().strftime("%H:%M:%S")

    # Read user information from user_info.csv
    user_info_path = './app_data/user_database/user_info.csv'
    with open(user_info_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            if row[0] == username:
                matric_number = row[1]
                course = row[2]
                break
        else:
            print(f"User {username} not found in user_info.csv")
            return
    
    # Create a new CSV file for attendance recording if it doesn't exist
    attendance_filename = f'attendance_{current_date}.csv'
    attendance_path = os.path.join('app_data/attendance_log', attendance_filename)

    # Check if the CSV file already exists
    file_exists = os.path.exists(attendance_path)

    fieldnames = ['Date', 'Timestamp', 'Username', 'Matric Number', 'Course']

    with open(attendance_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write attendance record to CSV
        writer.writerow({
            'Date': current_date,
            'Timestamp': timestamp,
            'Username': username,
            'Matric Number': matric_number,
            'Course': course
        })

    print(f"Attendance recorded for {username} on {current_date} at {timestamp}")


def search_database_image(username):
    user_db_dir = os.path.join(verification_img_dir, username)
    
    if not os.path.exists(user_db_dir):
        print(f"Directory '{user_db_dir}' does not exist")
        return []

    all_files = os.listdir(user_db_dir)
    return all_files

#
def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully")
    else:
        print(f"Directory '{directory}' already exists")



def recognize_face(db_faces, username, frame, x, y, w, h):
    face_captured = preprocess_image(os.path.join(input_img_dir, f'{username}.jpg'))
    
    val_count = 0
    for face in db_faces:
        full_dir = os.path.join(verification_img_dir, username)
        full_dir = preprocess_image(os.path.join(full_dir, face))
        verify_face(face_captured, full_dir, frame, username, x, y, w, h)
        val_count += 1
        print("------------------------------------------------------------------------------------------------")
        
        if val_count == validation_count:
            break

    if global_threshold/validation_count >= verification_threshold:
        record_attendance(username)
        print("Verified. Attendance taken")
    else:
        print("Unverified. Attendance not taken.")


def save_face_img(username, directory, face):
    img_path = os.path.join(directory, f'{username}.jpg')
    cv2.imwrite(img_path, face)
    print(f"Saved image: '{img_path}'")
    print("------------------------------------------------------------------------------------------------")


def face_valid_capture(capture, directory, username):
    db_img_files = search_database_image(username)
    
    while True:
        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_locate = face_classifier.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in face_locate:
            face_crop = frame[y:y + h, x:x + w]
            face = cv2.resize(face_crop, (400, 400))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            cap_cam = cv2.waitKey(1)
            if cap_cam == ord('c'):
                save_face_img(username, directory, face)
                recognize_face(db_img_files, username, frame, x, y, w, h)
                print("Press 'Esc' or 'q' to close camera...\n")
        
        cv2.imshow('Face Recognition', frame)
        esc_cam = cv2.waitKey(1)
        if esc_cam == ord('q') or esc_cam == 27:
            break
        
    capture.release()
    cv2.destroyAllWindows()


def main():
    global global_threshold
    
    while True:
        global_threshold = 0
        
        app_title = "Facial Verification Attendance System"
        print("------------------------------------------------------------------------------------------------")
        print(app_title.center(96))
        print("------------------------------------------------------------------------------------------------\n")
        username = str(input('Enter Your Name (Eg. Lan Yi Xian)\t: '))
        temp_dir = input_img_dir
        
        print("")
        mkdir(temp_dir)
        webcam = initialize_webcam()
        print("Press 'C' to perform face verification...\n")
        face_valid_capture(webcam, temp_dir, username)
        print("Press 'Esc' or 'q' again to quit or any other key to continue...\n\n\n")
        
        while True:
            key = keyboard.read_event(suppress=True).name
            if key == 'q' or key == 'esc':
                sys.exit()
            else:
                break


if __name__ == "__main__":
    main()
