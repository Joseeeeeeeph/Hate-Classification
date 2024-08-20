import os
import cv2
import multiprocessing
import numpy as np
import pandas as pd

class OCRItem():
    def __init__(self, array, name, labels):
        self.array = array
        self.name = name
        self.labels = labels

images_path = 'data/TextOCR/train_val_images/train_images'
standard_size = (100, 32)
p_batch_size = multiprocessing.cpu_count()
img_batch_size = 32
manager = multiprocessing.Manager()
ocr_data = manager.list()

image_list = pd.read_csv('data/TextOCR/img.csv', usecols=['file_name']).values.flatten()
r = np.shape(image_list)[0] % img_batch_size
r_batch = np.append(image_list[-r:], np.empty(shape=img_batch_size - r, dtype=object))
image_matrix  = np.append(image_list[:-r], r_batch).reshape(-1, img_batch_size)

annot_csv = pd.read_csv('data/TextOCR/annot.csv', usecols=['image_id', 'utf8_string'])
annotations_dict = annot_csv.groupby('image_id')['utf8_string'].apply(list).to_dict()

def get_batches(data, p_batch_size):
    total_batches = int(np.ceil(np.shape(data)[0] / p_batch_size))
    for i in range(total_batches):
        start = i * p_batch_size
        end = start + p_batch_size
        yield data[start:end]

def worker(files):
    img_batch = np.empty(shape=len(files), dtype=object)
    names = [name[6:] for name in files if name is not None]
    imgs = [cv2.imread(os.path.join(images_path, n), cv2.IMREAD_GRAYSCALE) for n in names]
    img_batch = zip(names, imgs)

    for img_pair in img_batch:
        name, img = img_pair
        if img is None: continue
        img = cv2.resize(img, standard_size)
        img_array = np.asarray(img).flatten()
        strings = annotations_dict.get(name[:-4], [])
        filtered_strings = [s for s in strings if s != '.']
        ocr_data.append(OCRItem(img_array, name, filtered_strings))

def parallel_process(p_batch):
    with multiprocessing.Pool(processes=p_batch_size) as pool:
        pool.map(worker, p_batch)

for p_batch in get_batches(image_matrix, p_batch_size):
    parallel_process(p_batch)

ocr_data = np.asarray(ocr_data)
np.random.shuffle(ocr_data)
partition = int(np.ceil(0.7 * np.shape(ocr_data)[0]))
ocr_training_data = ocr_data[:partition]
ocr_test_data = ocr_data[partition:]

alphabet = '0123456789abcdefghijklmnopqrstuvwxyz!"£$%^&*()_+-=[];:@#~,.<>?/'
nClasses = len(alphabet)