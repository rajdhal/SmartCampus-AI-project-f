from ultralytics import YOLO
from PIL import Image

model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

images = []
for i in range(4190, 4263):
    images.append(f'test2/Images/IMG_{i}.JPEG'.format(i))
    
# Run inference on 'bus.jpg'
results = model(images)  # results list

# Show the results
for i, r in enumerate(results):
    im_array = r.plot()  # plot a BGR numpy array of predictions
    im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    filename = f'test2/Results/results_{i + 4190}.jpg'
    im.save(filename)  # save image