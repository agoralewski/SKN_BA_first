# Imports
from config import main_config as mc  # user specific config
import pandas as pd
import numpy as np
import os
import datetime
import json
from tqdm import tqdm, tnrange  # trange(i) is a special optimised instance of tqdm(range(i))
from PIL import Image, ImageDraw


# Imports for the CNN
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Parameters
filelocation = mc.DATA_SET_DIR_PATH
nb_training_cases_per_cat = 100
RunForSelectedCategories = True


def create_dummy_submission(dataset_location, num_to_class):
    """ creates submission with random chosen classes assigned to test drawings """
    test_simplified_df = pd.read_csv(dataset_location + "test_simplified.csv")
    submission = test_simplified_df[['key_id']]
    # draw three ids of classes
    ids = np.random.randint(0, len(num_to_class), 3)  # TODO: here is sth wrong, maybe sb fix it:) hint: apple apple apple
    # translate ids to classes names joined by space
    submission['word'] = " ".join([num_to_class[k] for k in ids if k in num_to_class])
    submission.to_csv('submissions/submission_{}.csv'.format(datetime.date.today()),
                      index=False,
                      # header=['key_id', 'word']
                      )


def create_bitmap(input_json):
    image = Image.new("P", (256, 256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in json.loads(input_json):  # ast.literal_eval(input_json):
        for i in range(len(stroke[0])-1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i+1],
                             stroke[1][i+1]],
                            fill=0, width=5)
    # image = image.resize((imheight, imwidth))
    return np.array(image)/255.


def draw_raw_data_first_try():
    """ first try analyze raw data - drawings from Magda"""
    import ast
    import matplotlib.pyplot as plt
    test_raw = pd.read_csv(filelocation+"/test_raw.csv", index_col="key_id", nrows=100)

    # first 10 drawings
    first_ten_ids = test_raw.iloc[:10].index

    raw_images = [ast.literal_eval(lst) for lst in test_raw.loc[first_ten_ids, 'drawing']]
    for i in range(len(raw_images)):
        for x, y, t in raw_images[i]:
            plt.figure(figsize=(2, 1))
            plt.subplot(1, 2, 1)
            plt.plot(x, y, marker=".")
            plt.gca().invert_yaxis()


# Getting Data
train_files = os.listdir(filelocation+"train_simplified/")
num_to_class = {i: v[:-4].replace(" ", "_") for i, v in enumerate(train_files)}  # adds underscores

columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
selected_categories = ['airplane', 'axe', 'book', 'bowtie', 'cake', 'calculator']

if RunForSelectedCategories:
    train_files = [x+'.csv' for x in selected_categories]
    print(train_files)

train = pd.read_csv(filelocation + 'train_simplified/' + train_files[0], nrows=nb_training_cases_per_cat)
for file in train_files[1:]:
    print(file)
    sub_train = pd.read_csv(filelocation + 'train_simplified/' + file, nrows=nb_training_cases_per_cat)
    sub_train = sub_train[sub_train.recognized]
    train = train.append(sub_train)
    print(train.shape)
    print(train.shape[1])

print('!!! END !!! Train Test Size:', train.shape)
print(train.head(5))

# Filter for selected categories
# train_selected = train.loc[train['word'].isin(selected_categories)]

# Data transformation


# Model
# CNN
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(64, 64, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(train.shape[0], activation='softmax'))
model.summary()

# Compilation of the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Evaluation

# Export results to submission file
create_dummy_submission(filelocation, num_to_class)
