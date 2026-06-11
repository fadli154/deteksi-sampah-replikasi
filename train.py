from ultralytics import YOLO
import torch

def main():
    print("Torch:", torch.__version__)
    print("CUDA Available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")

    model.train(
        data="data.yaml",
        epochs=60,
        imgsz=640,
        batch=12,
        workers=4,
        cache=False,
        device=0
    )

if __name__ == "__main__":
    main()