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
files_location = mc.DATA_SET_DIR_PATH
nb_training_cases_per_cat = 300
run_selected_categories = True
SHUFFLED_CSV_NUMBER = 100
ROWS_PER_CLASS = 30000


# functions
def create_dummy_submission(files_location, num_to_class):
    """ creates submission with random chosen classes assigned to test drawings """
    test_simplified_df = pd.read_csv(files_location + "test_simplified.csv")
    submission = test_simplified_df[['key_id']]
    # draw three ids of classes
    ids = np.random.randint(0, len(num_to_class), 3)  # TODO: here is sth wrong, maybe sb fix it:) hint: apple apple apple
    # translate ids to classes names joined by space
    submission['word'] = " ".join([num_to_class[k] for k in ids if k in num_to_class])
    submission.to_csv('submissions/submission_{}.csv'.format(datetime.date.today()),
                      index=False,
                      # header=['key_id', 'word']
                      )


def create_bitmap(input_json, bitmap_height=256, bitmap_width=256):
    """
    Creates the bitmap of the drawing.
    Parameters
    ----------
    :param input_json: str
        A string that contains JSON array representing the simplified vector drawing.
        Array contains strokes. Each stroke is the two lists - first of Xs, second Ys
        [[[x0, x1, ...], [y0, y1, ...]], [[x0, x1, ...], [y0, y1, ...]]]
    :param bitmap_height: int
        bitmap height
    :param bitmap_width: int
        bitmap width
    Returns
    -------
    bitmap : array of floats
        The bitmap which contains the drawing - 0.0 is the background, 1.0 is the drawing line
    """
    image = Image.new("P", (256, 256), color=255)
    image_draw = ImageDraw.Draw(image)
    for stroke in json.loads(input_json):  # ast.literal_eval(input_json):
        for i in range(len(stroke[0]) - 1):
            image_draw.line([stroke[0][i],
                             stroke[1][i],
                             stroke[0][i + 1],
                             stroke[1][i + 1]],
                            fill=0, width=5)
    image = image.resize((bitmap_height, bitmap_width))
    return np.array(image) / 255.


def draw_raw_data_first_try(files_location):
    """ first try analyze raw data - drawings from Magda"""
    import ast
    import matplotlib.pyplot as plt
    test_raw = pd.read_csv(files_location + "/test_raw.csv", index_col="key_id", nrows=100)

    # first 10 drawings
    first_ten_ids = test_raw.iloc[:10].index

    raw_images = [ast.literal_eval(lst) for lst in test_raw.loc[first_ten_ids, 'drawing']]
    for i in range(len(raw_images)):
        for x, y, t in raw_images[i]:
            plt.figure(figsize=(2, 1))
            plt.subplot(1, 2, 1)
            plt.plot(x, y, marker=".")
            plt.gca().invert_yaxis()


def create_classes_dictionary(train_files_names):
    return {i: v[:-4].replace(" ", "_") for i, v in enumerate(train_files_names)}  # adds underscores


def read_train_simplified_csv(files_location, category, nrows=None, drawing_transform=False):
    """
    read train simplified csv of specific category
    Parameters
    ----------
    :param files_location: str
        data set location
    :param category: str
        category of data to be read
    :param nrows: int
        number of rows to read form one file
    :param drawing_transform: bool
        True if drawing have to be interpret as JSON
    Returns
    -------
    DataFrame : contains data from simplified csv file
    """
    file_name = category.replace("_", " ") + '.csv'
    df = pd.read_csv(os.path.join(files_location, 'train_simplified', file_name), nrows=nrows)
    if drawing_transform:
        df['drawing'] = df['drawing'].apply(json.loads)
    return df



# Getting Data
train_files = os.listdir(files_location + "train_simplified/")
print(train_files)
num_to_class = create_classes_dictionary(train_files)

columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
selected_categories = ['airplane', 'axe', 'book', 'bowtie', 'cake', 'calculator']

train = None
for i, category in enumerate(selected_categories) if run_selected_categories else num_to_class.items():
    print(category)
    if i == 0:
        train = read_train_simplified_csv(files_location, category, nb_training_cases_per_cat)
    else:
        train = train.append(read_train_simplified_csv(files_location, category, nb_training_cases_per_cat))
        # sub_train = sub_train[sub_train.recognized] # TODO lets think about this filtration
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
create_dummy_submission(files_location, num_to_class)



