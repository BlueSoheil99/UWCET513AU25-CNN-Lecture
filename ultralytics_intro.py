from ultralytics import YOLO

# load a pretrained model.
# models and their variants: https://docs.ultralytics.com/models/
# n:nano, s:small, m:medium, l:large, x:x-large

# model = YOLO('yolo11n.pt')  # downloads the model from ultralytics if not already downloaded
# model = YOLO('yolo11l.pt')
model = YOLO('fisheye_model.pt')  # the model trained from running the train_YOLO.ipynb with Google colab GPU

# run inference on the source
# save=True --> saves the output in runs/detect/predict

# results = model(source=1, show=True, conf=0.4) # opens webcam if available
# results = model(source=r'data/test1.mp4', show=True, conf=0.4, save=True)
results = model(source=r'data/test_fisheye.mp4', show=True, conf=0.4, save=True)

#object tracking:  https://docs.ultralytics.com/modes/track/
# results = model.track(r'data/test_fisheye.mp4', show=True, tracker="bytetrack.yaml")  # with ByteTrack

# you can use 'results' variable to process tracking/detection outputs
