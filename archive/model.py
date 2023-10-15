import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sys import stdout
import cv2
from scipy import ndimage
from IPython.display import clear_output


# Define if you want to download data from the original database or use the dataset one already provided and preprocessed
# Use:
# 'load': If you want to load the datase from the directory
# 'download': To download data from the database and process the images
dataset_load_method = 'load'

# Define if you want to save the dataset to a file
save_dataset = True

# Define if you want to load the trained classifiers from the directory
load_classifiers = False

# Define if you want to save the trained classifiers to a file
save_classifiers = True

# Define if you want to save classification test output to a file
save_results = True
if (save_results):
    result_output_file = open('result_output.txt','w') 

# Define if you want to print errors and warnings
enable_error_output = True


def print_percentage(prct, msg=None):
    if (prct > 100 or prct < 0):
        return
    clear_output(wait=True)
    if (msg == None):
        stdout.write("Progress: [")
    else:
        stdout.write(msg+" [")
    end = int(int(prct)/10)
    for i in range(0, end):
        stdout.write("=")
    for i in range(end, 10):
        stdout.write(" ")
    stdout.write("] "+str(prct)+"%")
    stdout.flush()



train_df = pd.read_csv("written_name_train_v2.csv", sep=",")
val_df = pd.read_csv("written_name_validation_v2.csv", sep=",")
test_df = pd.read_csv("written_name_test_v2.csv", sep=",")


def delborders(crop):
    cropf = ndimage.gaussian_filter(crop, 0.5)
    cropbin = (cropf<0.8)
    labeled, nr_objects = ndimage.label(cropbin)
    labels_to_delete = []
    for i in range(0, labeled.shape[1]):
        if (labeled[labeled.shape[0]-1][i] > 0):
            labels_to_delete.append(labeled[labeled.shape[0]-1][i])
    
    label_in_delete = False
    for x in range(0, labeled.shape[1]):
        for y in range(0, labeled.shape[0]):
            label_in_delete = False
            for l in range(0, len(labels_to_delete)):
                if (labeled[y][x] == labels_to_delete[l]):
                    label_in_delete = True
            
            if(label_in_delete):
                crop[y][x] = 1.0
    
    return crop


def getcrop(df, n):
    path = ""
    if df is train_df:
        path = "train_v2/train/"
    elif df is val_df:
        path = "validation_v2/validation/"
    elif df is test_df:
        path = "test_v2/test/"
    
    img = cv2.imread(path + df.iloc[n][0])
    imgh, imgw = img.shape[:-1]
    img_rgb = img.copy()
    template = cv2.imread('template(1).png')
    h, w = template.shape[:-1]

    res = cv2.matchTemplate(img_rgb, template, cv2.TM_CCOEFF_NORMED)
    threshold = .7
    loc = np.where(res >= threshold)

    if (len(loc[0])==0 and len(loc[1])==0):
        print("Template matching has failed for image: "+str(n))
        crop = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return crop, True
        
    return img, False



def gen_dataset(df, n:int=None):
    if (n == None):
        n = df.shape[0]
    data = []
    labels = []
    for i in range(0, n):
        crop, success = getcrop(df, i)
        if (success):
            data.append(crop)
            labels.append(df.iloc[i][1])
        else:
            if (enable_error_output):
                print("[WARNING] Template matching has failed for image: "+str(i))
        print_percentage((i*100/(n-1)), "Fetched "+str(i)+" images:")
    
    print_percentage(100, "Fetched "+str(n-1)+" images:")
    print("")
    print("Finished!")
    return data, labels


training_data, training_labels = gen_dataset(train_df, 20)
print("Training data: "+str(len(training_data)))
print("Training labels: "+str(len(training_labels)))

for i in range(0, len(training_data)):
    plt.imshow(training_data[i], cmap='gray')
    plt.show()