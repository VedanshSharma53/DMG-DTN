from ultralytics import YOLO
import cv2

# Load YOUR model
model = YOLO('runs/detect/train2/weights/best.pt')

# Option A: Predict on an Image
results = model.predict('image.png', save=True, show=True)

# Option B: Predict on a Video (Real-time)
# video_path = "dent.mp4" # Replace with your video path
# cap = cv2.VideoCapture(video_path)

# while cap.isOpened():
#     success, frame = cap.read()
#     if success:
#         # Run YOLO inference on the frame
#         results = model(frame)

#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break
#     else:
#         break

# cap.release()
# cv2.destroyAllWindows()
