from ultralytics import YOLO
from PIL import Image

model = YOLO('runs/detect/train/weights/best.pt')  # load a custom model

images = []
for i in range(4190, 4263):
    images.append(f'test1/Images/IMG_{i}.JPEG'.format(i))
    

for i, image_path in enumerate(images):
    result = model(image_path)  # predict on an image
    
    # Best Bounding Box
    bestBox = None
    bestConf = 0
    
    # OPen IMAGE
    im = Image.open(image_path)
    
    # Iterate over each box to find box with highest confidence rating
    for box in result[0].boxes:
        if box.conf.cpu().numpy()[0] > bestConf:
            bestConf = box.conf.item()
            bestBox = box
    
    # IF BESTBOX is not NONE get COORDS
    if bestBox is not None:
        x1 = bestBox.xyxy.cpu().numpy()[0][0]
        y1 = bestBox.xyxy.cpu().numpy()[0][1]
        x2 = bestBox.xyxy.cpu().numpy()[0][2]
        y2 = bestBox.xyxy.cpu().numpy()[0][3]
    
    im = im.crop((x1, y1, x2, y2))
    
    filename = f'test1/Results/results_{i + 4190}.jpg'
    im.save(filename)  # save image