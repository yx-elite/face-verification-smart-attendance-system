import cv2
import csv
import os
import uuid
import sys
import keyboard

# Load HAAR Classifier
face_classifier = cv2.CascadeClassifier('config/haarcascade_frontalface_default.xml')

def initialize_webcam():
    capture = cv2.VideoCapture(1)
    return capture


def create_csv(username, matric_num, course, csv_path):
    fieldnames = ['Username', 'Matric Number', 'Course']

    # Check if the CSV file already exists
    file_exists = os.path.exists(csv_path)

    with open(csv_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write header only if the file is newly created
        if not file_exists:
            writer.writeheader()

        # Write user information to CSV
        writer.writerow({'Username': username, 'Matric Number': matric_num, 'Course': course})

    print(f"User information saved to CSV: '{csv_path}'")



def mkdir(username, collection_path):
    if collection_path == 0:
        parent_dir = r'train_data\anchor'
    elif collection_path == 1:
        parent_dir = r'train_data\positive'
    else:
        raise ValueError("Invalid collection path")
    
    new_dir = os.path.join(parent_dir, username)
    
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
        print(f"\nDirectory '{username}' created successfully in '{parent_dir}'")
    else:
        print(f"\nDirectory '{username}' already exists in '{parent_dir}'")
    
    return new_dir


def save_face_img(new_dir, face):
    img_path = os.path.join(new_dir, f'{uuid.uuid1()}.jpg')
    cv2.imwrite(img_path, face)
    print(f"Saved image: '{img_path [5:]}'")

def face_capture(capture, new_dir, sample_size):
    sample_count = 0
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
                save_face_img(new_dir, face)
                sample_count += 1
        
        cv2.imshow("Camera", frame)
        esc_cam = cv2.waitKey(1)
        if esc_cam == ord('q') or esc_cam == 27 or sample_count >= sample_size:
            break

    capture.release()
    cv2.destroyAllWindows()


def main():
    while True:
        app_title = "Facial Data & Info Registration System"
        print("------------------------------------------------------------------------------------------------")
        print(app_title.center(96))
        print("------------------------------------------------------------------------------------------------\n")
        
        username = str(input("Enter your username (Eg. Lan Yi Xian)\t\t: "))
        matric_num = int(input("Enter your matric number (Eg. 151589)\t\t: "))
        course = str(input("Enter your course\t\t\t\t: "))
        sample_size = int(input("\nEnter number of face samples required\t\t: "))
        collection_path = int(input("Select directory ([0] Anchor / [1] Positive)\t: "))
        
        csv_path = r'app_data\user_database\user_info.csv'
        new_dir = mkdir(username, collection_path)
        create_csv(username, matric_num, course, csv_path)
        
        print("Web camera initializing...")
        webcam = initialize_webcam()
        
        print("\nData collection starting...")
        print("------------------------------------------------------------------------------------------------")
        
        face_capture(webcam, new_dir, sample_size)
        
        print("\n------------------------------------------------------------------------------------------------")
        print(f"{sample_size} face samples collected and stored in the directory: '{new_dir}'")
        print("------------------------------------------------------------------------------------------------")
        print("Data collection completed")
        print("Press 'Esc' or 'q' to quit or any other key to continue...\n\n\n")
        
        while True:
            key = keyboard.read_event(suppress=True).name
            if key == 'q' or key == 'esc':
                sys.exit()
            else:
                break


if __name__ == "__main__":
    main()