import os
import cv2
import pickle
import numpy as np
from fastapi import FastAPI, UploadFile, File
import uvicorn
from ultralytics import YOLO
import time




from detection import model
from recognition import recognize_face_facenet,process_frame,save_best_face_images,verify_attendance
from gallery_module import load_gallery_facenet, GALLERY_DIR, gallery_file




def main():
    gallery_path = "D:/AI/AiAttendanceProject/Data/gallery_facenet1.pkl"
    video_path = "D:/download/7mins.mp4"
    yolo_model_path = "D:/AI/AiAttendanceProject/App/runs/detect/train/weights/best.pt"
    output_dir = "D:/AI/AiAttendanceProject/Data/best_faces"


    with open(gallery_path, "rb") as f:
        gallery = pickle.load(f)


    yolo_model = YOLO(yolo_model_path)


    best_face_records = {}    
    detection_history = {}      
    unknown_detected = [False]  # Khởi tạo biến theo dõi unknown


    start_time = time.time()
    tracking_duration = 420  


    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open video!")
        return


    while True:
        ret, frame = cap.read()
        if not ret:
            break


        output_frame = process_frame(frame, yolo_model, gallery, best_face_records, detection_history, unknown_detected)


        elapsed_time = time.time() - start_time
        remaining_time = max(0, tracking_duration - elapsed_time)
        time_text = f"Remaining time: {int(remaining_time)} seconds"
        cv2.putText(output_frame, time_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow("Face Recognition", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


        if elapsed_time >= tracking_duration:
            break


    cap.release()
    cv2.destroyAllWindows()


    # Kiểm tra và in cảnh báo nếu có unknown
    if unknown_detected[0]:
        print("Cảnh báo: Có người không xác định trong video.")


    saved_paths = save_best_face_images(best_face_records, output_dir)


    print("\n--- Recognition results (raw) ---")
    print(f"Total recognized persons: {len(best_face_records)}")
    sorted_persons = sorted(best_face_records.items(), key=lambda x: x[1]['distance'])
    for name, data in sorted_persons:
        print(f"Name: {name}, Best Distance: {data['distance']:.4f}, Saved at: {saved_paths.get(name, 'Not saved')}")


    final_attendance = verify_attendance(detection_history, gallery,
                                         similarity_threshold=0.65, vote_threshold=0.7)
    print("\n--- Final Attendance ---")
    for name, vote_ratio in final_attendance.items():
        print(f"{name} confirmed with vote ratio: {vote_ratio:.2f}")


    print(f"\nBest face images saved to: {output_dir}")


    return best_face_records, final_attendance


if __name__ == "__main__":
    main()
