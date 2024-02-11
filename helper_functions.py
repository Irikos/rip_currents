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
# from sklearn.model_selection import KFold
import time
import cv2
import glob
import shutil
import os
import cv2
import os
import cv2
import re
from os.path import exists

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
        image = cv2.imread(path + img_path)
        images.append(image)
    return images

### Return a list of tuples (image, image_name)
def read_images2(path, format=".png", sorted=False):
    images_path = read_images_path(path, format, sorted)
    images = []
    for img_path in images_path:
        image = cv2.imread(path + img_path)
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
        '.avi': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        '.mp4': 0x7634706d
    }   
    destination = str(destination_folder + video_name)
    print(destination)
    video_output = cv2.VideoWriter(
        destination,
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
        '.avi': cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        '.mp4': 0x7634706d
    }   
    destination = str(destination_folder + video_name)
    print(destination)
    video_output = cv2.VideoWriter(
        destination, 
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
    cap = cv2.VideoCapture(video_path)

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
        


def rename_files_from_roboflow(path, extension):
    """
    Renames files in a given directory by removing a specific pattern and appending a new extension.

    This function is particularly useful for files downloaded from Roboflow, where the original 
    extension is appended at the end of the filename. It searches for files containing the pattern 
    '_jpg.rf' and replaces it with the specified extension.

    Parameters:
        path (str): Directory containing the files to be renamed.
        extension (str): New extension to be appended after removing the pattern.

    The function reports the number of files renamed and notifies about files not matching the pattern.

    Usage:
        rename_files_from_roboflow("path/to/folder", ".jpg")  # For images
        rename_files_from_roboflow("path/to/folder", ".txt")  # For label files
    """ 
    counter = 0
    for filename in glob.glob(os.path.join(path, '*.*')):
        if "_jpg.rf" in filename:
            os.rename(filename, filename.split("_jpg.rf")[0] + extension)
            counter += 1
        else:
            print("File " + filename + " does not contain '_jpg.rf'")
    print("Renamed " + str(counter) + " files")



def change_file_extension(path, old_extension, new_extension):
    """" 
    Change all file extensions in a folder from jpg to txt.
    Use in case you changed the file extension by mistake
    Usage: change_file_extension("path/to/folder", ".jpg", ".txt")
    """
    counter = 0
    for filename in glob.glob(os.path.join(path, '*.*')):
        if old_extension in filename:
            os.rename(filename, filename.split(old_extension)[0] + new_extension)
            counter += 1
        else:
            print("File " + filename + " does not contain " + old_extension)
    print("Changed " + str(counter) + " files")


def draw_annotations(image_path, label_path, output_path='', alpha=0.4, filled=False, border_thickness=2):
    """
    Draws bounding boxes and polygons on an image based on YOLO format annotations.

    Parameters:
        image_path (str): Path to the input image.
        label_path (str): Path to the label file with YOLO format annotations.
        output_path (str): Path for saving the output image.
        alpha (float): Transparency factor for filled shapes (default 0.4). Effective only if 'filled' is True.
        filled (bool): If True, draws filled shapes with transparency. If False, draws only the borders.
        border_thickness (int): Thickness of the borders (default 2).

    Annotations in the label file should be in YOLO format:
    - For bounding boxes: <object_class> <x_center> <y_center> <width> <height>
    - For polygons: <object_class> <x1> <y1> ... <xn> <yn>

    Note: Object class '0' is for bounding boxes, '1' is for polygons. Coordinates are normalized to the image dimensions.
    """
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    overlay = image.copy()

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            object_class = int(line[0])

            if object_class == 0:  # Bounding box
                x_center = float(line[1]) * width
                y_center = float(line[2]) * height
                w = float(line[3]) * width
                h = float(line[4]) * height
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)

                if filled:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=cv2.FILLED)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=border_thickness)

            elif object_class == 1:  # Polygon
                points = []
                for i in range(1, len(line), 2):
                    points.append((int(float(line[i]) * width), int(float(line[i + 1]) * height)))
                points = np.array([points], dtype=np.int32)

                if filled:
                    cv2.fillPoly(overlay, [points], (0, 0, 255))
                cv2.polylines(image, [points], True, (0, 0, 255), thickness=border_thickness)

    if filled:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    if output_path != '':
        cv2.imwrite(output_path, image)
    else:
        return image


def draw_annotations_on_image(image, label_path, output_path='', alpha=0.4, filled=False, border_thickness=2):
    """
    Draws bounding boxes and polygons on an image based on YOLO format annotations.

    Parameters:
        image_path (str): Path to the input image.
        label_path (str): Path to the label file with YOLO format annotations.
        output_path (str): Path for saving the output image.
        alpha (float): Transparency factor for filled shapes (default 0.4). Effective only if 'filled' is True.
        filled (bool): If True, draws filled shapes with transparency. If False, draws only the borders.
        border_thickness (int): Thickness of the borders (default 2).

    Annotations in the label file should be in YOLO format:
    - For bounding boxes: <object_class> <x_center> <y_center> <width> <height>
    - For polygons: <object_class> <x1> <y1> ... <xn> <yn>

    Note: Object class '0' is for bounding boxes, '1' is for polygons. Coordinates are normalized to the image dimensions.
    """
    height, width, channels = image.shape
    overlay = image.copy()

    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            object_class = int(line[0])

            if object_class == 0:  # Bounding box
                # YOLO to VOC format, but not used because Roboflow changed coords type if you also have instance segmentation
                # x_center = float(line[1])
                # y_center = float(line[2])
                # w = float(line[3])
                # h = float(line[4])
                # x1 = int((x_center - (w / 2)) * width)
                # y1 = int((y_center - (h / 2)) * height)
                # x2 = int((x_center + (w / 2)) * width)
                # y2 = int((y_center + (h / 2)) * height)
                x1 = int(float(line[1]) * width)
                y1 = int(float(line[2]) * height)
                x2 = int(float(line[5]) * width)
                y2 = int(float(line[6]) * height)


                if filled:
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), thickness=cv2.FILLED)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness=border_thickness)

            elif object_class == 1:  # Polygon
                points = []
                for i in range(1, len(line), 2):
                    points.append((int(float(line[i]) * width), int(float(line[i + 1]) * height)))
                points = np.array([points], dtype=np.int32)

                if filled:
                    cv2.fillPoly(overlay, [points], (0, 0, 255))
                cv2.polylines(image, [points], True, (0, 0, 255), thickness=border_thickness)

    if filled:
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    if output_path != '':
        cv2.imwrite(output_path, image)
    else:
        return image

def create_subfolders_with_annotations(videos_folder, annotations_folder, destination_folder):
    """
    Copies annotation files to corresponding video frame subfolders, ensuring each frame is annotated.

    This function iterates over video files in a specified folder, creates subfolders for each video in a destination folder, and then copies annotation files from an annotations folder to these subfolders. Each video frame is assigned an annotation file based on a calculated distribution.

    Parameters:
        videos_folder (str): Directory containing video files.
        annotations_folder (str): Directory containing annotation files.
        destination_folder (str): Directory where subfolders for each video will be created.

    For each video, the function:
    - Creates a subfolder named after the video.
    - Calculates the number of frames in the video.
    - Determines the number of frames that each annotation file will cover.
    - Copies each annotation file to the appropriate subfolder, renaming it to match the frame index.

    The function assumes that video file names start with 'DJI_' and annotation files end with '.txt'. It ensures that all frames are annotated, even if the division of frames to annotation files is uneven.

    Example:
        create_subfolders_with_annotations('path/to/videos', 'path/to/annotations', 'path/to/destination')
    """
    # Get the list of video files starting with "DJI_" or "VIDEO-"
    video_files = [file for file in os.listdir(videos_folder) if (file.startswith("DJI_") or file.startswith("VIDEO-"))]

    # Iterate over each video file
    for video_file in video_files:
        video_path = os.path.join(videos_folder, video_file)
        video_name = os.path.splitext(video_file)[0]
        subfolder_path = os.path.join(destination_folder, video_name)
        os.makedirs(subfolder_path, exist_ok=True)

        num_frames = get_num_frames(video_path)
        video_annotations_folder = os.path.join(annotations_folder, video_name)
        if not os.path.exists(video_annotations_folder):
            print(f"Annotation folder for {video_file} does not exist")
            continue
        annotation_files = [file for file in os.listdir(video_annotations_folder) if file.upper().endswith(".TXT")]

        total_initial_annotations = max([int(file.split("-")[-1].split(".")[0]) for file in annotation_files])
        # Calculate the number of frames each annotation file should cover
        frames_per_annotation = num_frames / float(total_initial_annotations)

        # Use a variable to track the current frame count as a floating point
        current_frame_coverage = frames_per_annotation
        current_annotation_index = 0

        # Iterate over each frame in the video
        print(f"Processing {video_file}...")
        for frame_index in range(num_frames):
            # Update the annotation index based on the current_frame_coverage
            if frame_index >= current_frame_coverage:
                current_annotation_index += 1 
                current_frame_coverage = (current_annotation_index + 1) * frames_per_annotation
                # Ensure the annotation index does not exceed the number of available files
                current_annotation_index = min(current_annotation_index, len(annotation_files) - 1)

            annotation_file = f"{video_name}_MP4-{current_annotation_index}.txt"
            
            # Ensure we increment the annotation index only when the coverage exceeds the threshold
            if frame_index >= current_frame_coverage:
                current_annotation_index += 1
                
            annotation_source = os.path.join(video_annotations_folder, annotation_file)
            annotation_destination = os.path.join(subfolder_path, f"{video_name}-{frame_index}.txt")
            if not os.path.exists(annotation_source):
                # print(f"Annotation file {annotation_file} does not exist")
                open(annotation_destination, 'a').close()
            else: 
                shutil.copy(annotation_source, annotation_destination)

def get_num_frames(video_path):
    # Your existing code to get the number of frames in a video
    # return len(hf.video_to_frames(video_path))
    
    cap = cv2.VideoCapture(video_path)
    property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
    length = int(cv2.VideoCapture.get(cap, property_id))
    return length


def process_videos_to_frames(videos_folder, destination_folder):
    # Get the list of video files starting with "DJI_"
    video_files = [file for file in os.listdir(videos_folder) if (file.startswith("DJI_") or file.startswith("VIDEO-"))]

    # Iterate over each video file
    for video_file in video_files:
        # Create a subfolder with the video name in the destination folder
        video_name = os.path.splitext(video_file)[0]
        subfolder_path = os.path.join(destination_folder, video_name)
        os.makedirs(subfolder_path, exist_ok=True)

        # Read the video file
        video_path = os.path.join(videos_folder, video_file)

        frames = video_to_frames(video_path)
        
        # Get the total number of frames in the video
        num_frames = len(frames)

        # Iterate over each frame in the video
        print(f"Processing {video_file}...")
        for frame_index in range(len(frames)):
            # Save the processed frame in the subfolder
            frame_path = os.path.join(subfolder_path, f"{video_name}_{frame_index}.jpg")
            cv2.imwrite(frame_path, frames[frame_index])

    
def process_videos(videos_folder, annotations_folder, destination_folder):
    # Get the list of video files starting with "DJI_"
    video_files = [file for file in os.listdir(videos_folder) if (file.startswith("DJI_") or file.startswith("VIDEO-"))]

    # Iterate over each video file
    for video_file in video_files:
        print(f"Processing {video_file}...")
        # Create a subfolder with the video name in the destination folder
        video_name = os.path.splitext(video_file)[0]

        # Read the video file   
        video_path = os.path.join(videos_folder, video_file)
        cap = cv2.VideoCapture(video_path)

        # Get the video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a VideoWriter object to save the processed video
        output_path = os.path.join(destination_folder, f"{video_name}_processed.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Get the list of annotation files for the current video
        annotation_folder_path = os.path.join(annotations_folder, video_name)
        annotation_files = os.listdir(annotation_folder_path)

        # Iterate over each frame in the video
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Get the annotation file for the current frame
            annotation_file = f"{video_name}-{frame_index}.txt"
            # annotation_file = f"{video_name}_MP4-{frame_index}.txt"
            if annotation_file not in annotation_files:
                print(f"Annotation file {annotation_file} does not exist")
                frame_index += 1
                continue
            annotation_file_path = os.path.join(annotation_folder_path, annotation_file)

            # Apply the function to draw bounding boxes on the frame
            processed_frame = draw_annotations_on_image(frame, annotation_file_path, alpha=0.4, filled=False, border_thickness=2)

            # Write the processed frame to the output video
            out.write(processed_frame)

            frame_index += 1

        # Release the video capture and writer objects
        cap.release()
        out.release()

        print(f"Processed {video_file} and saved to {output_path}")


def move_files_by_prefix(folder_path):
    # Get all files in the folder
    files = os.listdir(folder_path)
    
    # Create a dictionary to store files by prefix
    file_dict = {}
    
    # Iterate over each file
    for file in files:
        file_name_without_extension, file_extension = os.path.splitext(file)
        new_file_name = file_name_without_extension.upper() + file_extension
        file = new_file_name  # Convert file name to uppercase
        if "_MP4" in file:
            # Get the prefix of the file
            prefix = file.split("_MP4")[0]
            # Check if the prefix already exists in the dictionary
            if prefix in file_dict:
                # If the prefix exists, append the file to the list of files with the same prefix
                file_dict[prefix].append(file)
            else:
                # If the prefix does not exist, create a new list with the file
                file_dict[prefix] = [file]
        
    # Iterate over the dictionary and move the files to the corresponding folders
    for prefix, files in file_dict.items():
        # Create a folder with the prefix name
        folder_name = os.path.join(folder_path, prefix)
        os.makedirs(folder_name, exist_ok=True)
        
        # Move each file to the corresponding folder
        for file in files:
            file_path = os.path.join(folder_path, file)
            destination_path = os.path.join(folder_name, file)
            shutil.move(file_path, destination_path)
    
    print("Files moved successfully.")

    # Example usage
    # folder_path = "/path/to/folder"
    # move_files_by_prefix(folder_path)

def rename_files_to_uppercase(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isfile(os.path.join(folder_path, filename)):
            new_filename = filename.upper()
            os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_filename))
    print("Done renaming files to uppercase.")



def rename_files_format_number(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            old_name = os.path.join(root, file)
            new_name = re.sub(r'-(\d+)', lambda match: '-' + str(int(match.group(1))), old_name)
            os.rename(old_name, new_name)


def rename_file_extensions_to_lowercase(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_name, file_extension = os.path.splitext(file)
            new_file_name = file_name + file_extension.lower()
            new_file_path = os.path.join(root, new_file_name)
            os.rename(file_path, new_file_path)