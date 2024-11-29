from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/john/yolo-datasets/roboflow-universe-model/data.yaml", epochs=400, imgsz=640)
