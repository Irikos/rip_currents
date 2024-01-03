### All the helper functions I have used when working with the rip currents datasets and models. ###
### I will try to create a separate file in each paper's folder with the ones that you need when importing the data. No promises, though, and you can find them all here if needed ###


print("Hello, world!")
print("Actual file with the helper functions that can be imported.")

### IMPORTS ###
print("Importing libraries...")
from PIL import Image, ImageDraw
import os
import PIL
import numpy as np
from sklearn.model_selection import KFold
import time
import cv2 as cv
import glob

print("Done importing libraries.")

#### HELPER FUNCTIONS for PASCAL VOC to YOLO annotations format #### 

# INPUT: 
#   data_folder: string, path to folder with the images
#   pascal_voc_labels_file: string, path to file with the labels of the images in PASCAL VOC format
#   yolo_annotations_folder: string, path to folder where the yolo annotations will be stored
#   verbose: boolean, if true, print the progress
# OUTPUT: 
#   writes the yolo annotations files in the yolo_annotations_folder
def pascal_voc_to_yolo(data_folder, pascal_voc_labels_file, yolo_annotations_folder, verbose=False):
    if (verbose==True):
        print("Started converting Pascal VOC to YOLO annotations.")
    # Read labels file to dictionary
    if (verbose==True):
        print("Step 1/4: Reading pascal annotations...")
    pascal_voc_annotations = read_pascal_annotations(pascal_voc_labels_file, verbose)
    # Add width and height to each of them by reading the images
    if (verbose==True):
        print("Step 2/4: Updating annotations with dimensions...")
    pascal_voc_annotations = update_annotations_with_dimensions(data_folder, pascal_voc_annotations, verbose)
    # Convert labels to yolo format
    if (verbose==True):
        print("Step 3/4: Converting labels to yolo format...")
    yolo_annotations = pascal_dict_to_yolo_dict(pascal_voc_annotations, verbose)
    # Write to yolo annotations folder
    if (verbose==True):
        print("Step 4/4: Writing to yolo annotations folder...")
    write_yolo_annotation_to_files(yolo_annotations, yolo_annotations_folder, verbose)
    return 0

# INPUT:
#   annotation_files: list, with files with the annotations in PASCAL VOC format
#   verbose: boolean, if true, print the progress
# OUTPUT:
#   pascal_annotations: dictionary, with the annotations in PASCAL VOC format
def read_pascal_annotations(annotations_file, verbose=False):
    if (verbose==True):
        i = 1
    try:
        if (verbose==True):
            with open(annotations_file, 'r') as f:
                file = open(annotations_file, "r")
                print("Reading file ", annotations_file)
                length = len(file.readlines())
                file.close()
        with open(annotations_file, 'r') as f:
            file = open(annotations_file, "r")
            new_line = file.readline()
            pascal_annotations = {}
            i = 1
            while (new_line):
                annotation = read_annotation(new_line)
                pascal_annotations[annotation['filename']] = annotation
                new_line = file.readline()
                if (verbose==True):
                    print("Read annotation ", i, " of ", length)
                    i += 1
            file.close()
            return pascal_annotations
    
    except IOError:
        print ("Could not read file: ", annotations_file)
        exit()

# INPUT: 
#   data_folder: string, path to the folder with the images
#   annotations: dictionary, with the annotations in PASCAL VOC format
#   verbose: boolean, if true, print the progress
# OUTPUT:
#  annotations: dictionary, with the annotations in PASCAL VOC format with added height and widht of the image
def update_annotations_with_dimensions(data_folder, annotations, verbose=False):
    if (verbose==True):
        i = 1
        length = len(annotations)
    for image_name in annotations:
        annotations[image_name].update(add_width_and_height_to_annotation(annotations[image_name], data_folder))
        if (verbose==True):
            print("Updated annotation (with dimensions) ", i, " of ", length)
            i += 1
    return annotations

# INPUT:
#   annotations: dictionary, representing 1 (one) annotation in PASCAL VOC format
#   data_folder: string, path to the folder with the images
# OUTPUT:
#   annotation: dictionary, representing 1 (one) annotation in PASCAL VOC format with added height and widht of the image
def add_width_and_height_to_annotation(annotation, data_folder):
    try:
        width, height = read_image_width_and_height(os.path.join(data_folder, annotation['filename']))
        annotation['width'] = str(width)
        annotation['height'] = str(height)
    except:
        print("Error reading image", annotation['filename'])
    return annotation

# INPUT:
#   source_annotation: list, representing 1 (one) annotation in PASCAL VOC format    
#   source annotation format: <image_name.extension>, xmin, ymin, xmax, ymax, class
#   verbose: boolean, if true, print the progress
# OUTPUT: 
#   target_annotation: dictionary, representing 1 (one) annotation in YOLO format
#   target_annotation format: <class>, <b_center_x>, <b_center_y>, <b_width>, <b_height>
def read_annotation(source_annotation, verbose=False):
    items = source_annotation.split(',')
    target_annotation = {}
    target_annotation['filename'] = items[0]
    target_annotation['xmin'] = items[1]
    target_annotation['ymin'] = items[2]
    target_annotation['xmax'] = items[3]
    target_annotation['ymax'] = items[4]
    target_annotation['class'] = str(items[5]).strip() # as it is the last item, it has a newline
    if (verbose==True):
        print("Read annotation: ", target_annotation)
    return target_annotation

# INPUT:
#   image_name: string, name of the image
# OUTPUT:
#   width: int, width of the image
#   height: int, height of the image
def read_image_width_and_height(image_path):
    image = PIL.Image.open(image_path)
    width, height = image.size
    return width, height

# INPUT:
#   annotation_classes: string, hardcoded for now
# OUTPUT:
#   converted_clas: string, converted class from string to number (as string)
def convert_classes(annotation_class):
    converted_class = 0
    if annotation_class=='rip':
        converted_class = '0'
    return converted_class

# INPUT:
#   pascal_annotation: dictionary, with the annotations in PASCAL VOC format
# OUTPUT:
#   yolo_annotation: list, with the annotations in YOLO format
def pascal_to_yolo(pascal_annotation):
    # bbox coordonates for yolo format
    b_center_x = (float(pascal_annotation['xmin']) + float(pascal_annotation['xmax'])) / 2
    b_center_y = (float(pascal_annotation['ymin']) + float(pascal_annotation['ymax'])) / 2
    b_width = (float(pascal_annotation['xmax']) - float(pascal_annotation['xmin']))
    b_height = (float(pascal_annotation['ymax']) - float(pascal_annotation['ymin']))
    # Normalise the coordinates by the dimensions of the image
    b_center_x /= float(pascal_annotation['width'])
    b_center_y /= float(pascal_annotation['height'])
    b_width /= float(pascal_annotation['width'])
    b_height /= float(pascal_annotation['height'])

    b_center_x = float(round(b_center_x, 3))
    b_center_y = float(round(b_center_y, 3))
    b_width = float(round(b_width, 3))
    b_height = float(round(b_height, 3))
    
    yolo_annotation = " "
    yolo_items = [str(convert_classes(pascal_annotation["class"])), str(b_center_x), str(b_center_y), str(b_width), str(b_height)]
    yolo_annotation = yolo_annotation.join(yolo_items)
    return yolo_annotation

# INPUT:
#   pascal_dictionary: dictionary, with the annotations in PASCAL VOC format
#   verbose: boolean, if true, print the progress
# OUTPUT:
#   yolo_dict: dictionary, with the annotations in YOLO format
def pascal_dict_to_yolo_dict(pascal_dict, verbose=False):
    if (verbose==True):
        i = 1
        length = len(pascal_dict)
    yolo_dict = {}
    for image_name in pascal_dict:
        yolo_dict[image_name] = pascal_to_yolo(pascal_dict[image_name])
        if (verbose==True):
            print("Converted annotation (to yolo format)", i, " of ", length)
            i += 1
    return yolo_dict

# INPUT:
#   yolo_annotation: list, with the annotations in YOLO format
#   path: string, path to the folder where we want to save the YOLO annotations
#   verbose: boolean, if true, print the progress
# OUTPUT:
#   writes the files with the annotations at the file path
def write_yolo_annotation_to_files(yolo_annotations, path, verbose=False):
    if (verbose==True):
        i = 1
        length = len(yolo_annotations)
    if (not os.path.exists(path)):
        os.makedirs(path)
    for annotation_file in yolo_annotations:
        filename = annotation_file.split('.')[0] + '.txt'
        file_full_path = os.path.join(path, filename)
        file = open(file_full_path, "w+")
        file.write(yolo_annotations[annotation_file])
        if (verbose==True):
            print("Wrote yolo annotation to file ", i, " of ", length)
            i += 1
        file.close()
    print("Finished writing yolo annotations to files.")

###############################################################################################################################


################## HELPER FUNCTIONS FOR FILE READING FOR YOLO #############################################################################

# LOAD IMAGES PATH INTO AN ARRAY
# INPUT:
#   path - path to folder with images
#   sorted - boolean that applies sort to files. Only works if all the files contain numbers
# OUTPUT:
#   files - an array with all the images paths
def read_images_path(path, format=".jpg", sorted=False):
    files = []
    for file in os.listdir(path):
        if file.endswith(format) or file.endswith(format.upper()):
            files.append(file)
    if (sorted==True):
        files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return files

# LOAD THE ANNOTATION FILES INTO AN ARRAY
# INPUT: 
#   path - path to folder with annotations. Expected yolo format, with one txt file corresponding to one image
# OUTPUT:
#   files - an array with all the annotations paths
def read_annotations(path, format=".txt"):
    files = []
    for file in os.listdir(path):
        if file.endswith(format) or file.endswith(format.upper()):
            files.append(file)
    return files


### WRITE THE YOLO CUSTOM DATASET.YAML. Wrote with kfold validation in mind, but useful in general. ###
# INPUT:
#   yaml_path: dataset.yaml path
#   dataset_path: path to the dataset folder
#   train_file: path to file with training images paths (relative to dataset folder)
#   validation_file: path to file with validation images paths (relative to dataset folder)
#   test_file: path to file with test images paths (relative to dataset folder). Can be empty.
#   classes: array of class names to be written in the dataset (note: now classes are hardcoded)
def write_yolo_dataset_yaml(classes, yaml_path, dataset_path, train_file = "./training.txt", validation_file = './validation.txt', test_file = ''):
    file = open(yaml_path, "w+")

    file.write("path: " + dataset_path + "\n")
    file.write("train: " + train_file + "\n") # relative to dataset_path
    file.write("val: " + validation_file + "\n")
    file.write("test: " + test_file + "\n") # not used yet
    file.write("nc: " + str(len(classes)) + "\n")
    file.write("names: " + str(classes) + "\n")
    file.close()

  

# RIPS_FOLDER = "../datasets/rip_currents/training_data/with_rips" 
# PASCAL_VOC_LABELS = "../datasets/rip_currents/training_data/with_rips/data_labels.txt"
# YOLO_FOLDER = "../datasets/rip_currents/training_data/yolo_annotations"
# pascal_voc_to_yolo(RIPS_FOLDER, PASCAL_VOC_LABELS, YOLO_FOLDER, verbose=True)


### END OF HELPER FUNCTIONS ###


### USEFUL PATHS ###
TRAINING_PATH = "../datasets/rip_currents/training_data"  
RIPS_FOLDER = "../datasets/rip_currents/training_data/images" 
PASCAL_VOC_LABELS = "../datasets/rip_currents/training_data/with_rips/data_labels.txt"
YOLO_FOLDER = "../datasets/rip_currents/training_data/labels"
### END ###

### RUNNING FUNCTIONS ###
# converting VOC to YOLO annotations
# RIPS_FOLDER - folder with images of rip currents
# PASCAL_VOC_LABELS - .txt file with annotations for each rip current images in VOC format
# YOLO_FOLDER - folder where to save the yolo annotations files
# hf.pascal_voc_to_yolo(RIPS_FOLDER, PASCAL_VOC_LABELS, YOLO_FOLDER, verbose=True)


### CROSS VALIDATION ###


# DATASET_PATH = "../datasets/rip_currents/training_data"
# TRAIN_FILE_PATH = "../datasets/rip_currents/training_data/training.txt"
# VALIDATION_FILE_PATH = "../datasets/rip_currents/training_data/validation.txt"
# DATASET_YAML_PATH = "../datasets/rip_currents/training_data/rip_currents.yaml"
# TEST_FILE_PATH = ""
# classes = ["rip_current"]



# X = np.array(read_images_path(RIPS_FOLDER))
# y = read_annotations(YOLO_FOLDER)

### TRAINING FUNCTIONS ###
### WRITTEN FOR JUPYTER NOTEBOOK, NOT .py FILES ###
def train_yolo_kfold(X, y, train_file_path, validation_file_path, dataset_yaml_path, dataset_path, model_log_name, classes, model_weights="", img_size=640, batch_size=128, epochs=2000, workers=24, verbose=True, splits=10):
    i = 0
    kf = KFold(n_splits=splits)
    for train_index, validation_index in kf.split(X):
        if (verbose):
            print("Starting fold no: " + str(i))

        start_time = time.time()

        X_train, X_validation = X[train_index], X[validation_index]

        file = open(train_file_path, "w+")
        for item in X_train:
            file.write("./images/" + item + "\n")
        file.close()

        file = open(validation_file_path, "w+")
        for item in X_validation:
            file.write("./images/" + item + "\n")
        file.close()
        write_yolo_dataset_yaml(classes, dataset_yaml_path, dataset_path)
        
        current_iteration = model_log_name + str(i)
        
        # !python ./train.py --img $img_size --cfg ./models/yolov5s.yaml --hyp ./data/hyps/hyp.scratch-low.yaml --batch $batch_size --epochs $epochs --data $dataset_yaml_path --weights "yolov5s.pt" --workers $workers --name $current_iteration
        
        e = int(time.time() - start_time)
        if (verbose):
            print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
            print("Ended fold no: " + str(i) + "\n")
        i += 1


### Return a list of images
def read_images(path, format=".png", sorted=False):
    images_path = read_images_path(path, format, sorted)
    images = []
    for img_path in images_path:
        image = cv.imread(path + img_path)
        images.append(image)
    return images

### Return a list of tuples (image, image_name)
def read_images(path, format=".png", sorted=False):
    images_path = read_images_path(path, format, sorted)
    images = []
    for img_path in images_path:
        image = cv.imread(path + img_path)
        images.append((image, img_path.split('.')[0]))
    return images


### PROCESS FRAMES INTO VIDEOS ###
def frames_to_video(
        frames_folder, frames_format=".png", annotations_folder=None, 
        destination_folder="./", video_name="reconstructed.mp4",
        video_format=".mp4",
        fps=30, augmentations=None
        ):
    # frames = read_images_path(frames_folder, frames_format)
    frames = read_images(frames_folder, frames_format, sorted=True)
    h, w, _ = frames[0].shape
    # h, w, _ = frames[0].size # assumes all frames have the same shape
    # here we have the extensions and the fourcc for each of it
    video_extension_and_fourcc_dict = {
        '.avi': cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        '.mp4': 0x7634706d
    }   
    print(destination_folder + video_name)
    video_output = cv.VideoWriter(
        # destination_folder + video_name, 
        video_name,
        video_extension_and_fourcc_dict[video_format],
        fps,
        (w, h),
        True        
    )
    for frame in frames:
        video_output.write(frame)

    video_output.release()


### PROCESS FRAMES INTO VIDEOS ###
def frames_to_video2(
        frames, annotations_folder=None, 
        destination_folder="./", video_name="reconstructed.mp4",
        video_format=".mp4",
        fps=30, augmentations=None
        ):
    # frames = read_images_path(frames_folder, frames_format)
    h, w, _ = frames[0].shape
    # h, w, _ = frames[0].size # assumes all frames have the same shape
    # here we have the extensions and the fourcc for each of it
    video_extension_and_fourcc_dict = {
        '.avi': cv.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        '.mp4': 0x7634706d
    }   
    print(destination_folder + video_name)
    video_output = cv.VideoWriter(
        # destination_folder + video_name, 
        video_name,
        video_extension_and_fourcc_dict[video_format],
        fps,
        (w, h),
        True        
    )
    for frame in frames:
        video_output.write(frame)

    video_output.release()

### EXTRACT FRAMES FROM VIDEO ###
def video_to_frames(video_path):
    frames = []
    cap = cv.VideoCapture(video_path)

    if cap.isOpened() == False:
        print("Error opening video stream or file")
        return frames

    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            frames.append(frame)
        else:
            break
    cap.release()
    return frames



### RESIZE ALL IMAGES IN A FOLDER ###
# INPUT:
#   images_folder: path to folder containing the images to be resized
#   destination_folder: path to folder where the resized images are saved (automatically created if it does not exist)
#   width: destination width
#   height: desetination height
def resize_images(images_folder, destination_folder, width, height):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    images = os.listdir(images_folder)
    for image in images:
        if os.path.isfile(images_folder + image):
            im = Image.open(images_folder + image)
            file_name, file_ext = os.path.splitext(image)
            imResize = im.resize((width, height), Image.ANTIALIAS)
            imResize.save(destination_folder + file_name + ' resized.png', 'PNG', quality=90) 
            
### WRITE THE YOLO CUSTOM DATASET.YAML. Wrote with kfold validation in mind, but useful in general. ###
# INPUT:
#   yaml_path: dataset.yaml path
#   dataset_path: path to the dataset folder
#   train_file: path to file with training images paths (relative to dataset folder)
#   validation_file: path to file with validation images paths (relative to dataset folder)
#   test_file: path to file with test images paths (relative to dataset folder). Can be empty.
#   classes: array of class names to be written in the dataset (note: now classes are hardcoded)
DATASET_PATH = "/home/irikos/Work/datasets/rip_currents/training_data"
TRAIN_FILE_PATH = "/home/irikos/Work/datasets/rip_currents/training_data/training.txt"
VALIDATION_FILE_PATH = "/home/irikos/Work/datasets/rip_currents/training_data/validation.txt"
DATASET_YAML_PATH = "/home/irikos/Work/datasets/rip_currents/training_data/rip_currents.yaml"
TEST_FILE_PATH = ""

classes = ["rip_current"]


def kfold_split_yolo_dataset_yml(files_destination, dataset_path, annotations_path, classes, verbose=False):
    ## JUST TESTING ONCE ##
    X = np.array(read_images_path(dataset_path))
    kf = KFold(n_splits=10)
    i = 0
    for train_index, validation_index in kf.split(X):
        print("Starting fold no: " + str(i))
        start_time = time.time()
        X_train, X_validation = X[train_index], X[validation_index]

        file = open(files_destination + "/training_k" + str(i) + ".txt", "w+")
        for item in X_train:
            file.write(dataset_path + "/" + item + "\n")
        file.close()

        file = open(files_destination + "/validation_k" + str(i) + ".txt", "w+")
        for item in X_validation:
            file.write(dataset_path + "/" + item + "\n")
        file.close()
        write_yolo_dataset_yaml(classes, files_destination + "/rip_currents_k" + str(i) + ".yaml", dataset_path, files_destination + "/training_k" + str(i) + ".txt", files_destination + "/validation_k" + str(i) + ".txt")

        # current_iteration = "yoloV7_K_" + str(i)

        #!python /home/irikos/Work/yolov7/train.py --workers 24 --device 0 --batch-size 32 --data /home/irikos/Work/datasets/rip_currents/training_data/rip_currents.yaml --img 640 640 --cfg /home/irikos/Work/yolov7/cfg/training/yolov7-custom.yaml --weights /home/irikos/Work/yolov7/weights/yolov7.pt --name yolov7-custom --hyp /home/irikos/Work/yolov7/data/hyp.scratch.custom.yaml --name $current_iteration
        # !python ./train.py --img 640 --cfg ./models/yolov5s.yaml --hyp ./data/hyps/hyp.scratch-low.yaml --batch 128 --epochs 300 --data ../datasets/rip_currents/training_data/rip_currents.yaml --weights yolov5s.pt --workers 24 --name $current_iteration
        i += 1
        e = int(time.time() - start_time)
        print('{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
        print("Ended fold no: " + str(i) + "\n")
        


# parse all files in a folder and remove everything after the "_jpg.rf" part in the file name 
# (check if it exists first) and add its extension at the end
# Usage: rename_files("path/to/folder", ".txt")
# apply for both labels ('.txt') and images ('.jpg')

# Useful when for files downloaded from Roboflow, which append the extension at the end of the file name

def rename_files(path, extension):
    counter = 0
    for filename in glob.glob(os.path.join(path, '*.*')):
        if "_jpg.rf" in filename:
            os.rename(filename, filename.split("_jpg.rf")[0] + extension)
            counter += 1
        else:
            print("File " + filename + " does not contain '_jpg.rf'")
    print("Renamed " + str(counter) + " files")


# change all file extensions in a folder from jpg to txt.
# use in case you changed the file extension by mistake
# Usage: change_file_extension("path/to/folder", ".jpg", ".txt")

def change_file_extension(path, old_extension, new_extension):
    counter = 0
    for filename in glob.glob(os.path.join(path, '*.*')):
        if old_extension in filename:
            os.rename(filename, filename.split(old_extension)[0] + new_extension)
            counter += 1
        else:
            print("File " + filename + " does not contain " + old_extension)
    print("Changed " + str(counter) + " files")