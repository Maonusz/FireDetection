from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('C:\\Users\\ADMIN\\Desktop\\model_new\\Yolov8_FireDetect_Export\\best.pt')
# Export the model to ONNX format
model.export(format='onnx')
