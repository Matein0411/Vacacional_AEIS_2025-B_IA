from ultralytics import YOLO
import os


class Model:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    
    def predict(self, source):
        current_dir = os.getcwd()
        results = self.model.predict(source=source, show=True, save=True, save_dir=current_dir)
        return results


if __name__ == "__main__":
    model_path = "yolov8n (2).pt"  
    source = 0

    model = Model(model_path)
    results = model.predict(source)
    print(results)