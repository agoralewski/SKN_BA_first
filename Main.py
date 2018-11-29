# Imports
from config import main_config as mc  # user specific config
import pandas as pd
import numpy as np
import os
import datetime
import json
from tqdm import tqdm, tnrange  # trange(i) is a special optimised instance of tqdm(range(i))
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

# Imports for the CNN
#import tensorflow as tf
#from tensorflow import keras
#from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Parameters
FILES_LOCATION = mc.DATA_SET_DIR_PATH
nb_training_cases_per_cat = 300
run_selected_categories = True
SHUFFLED_CSV_NUMBER = 100
ROWS_PER_CLASS = 300


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


def create_shuffled_data(source_files_location, destination_files_location, num_to_class, shuffled_csv_number, rows_per_class=None):
    """
    function creates CSVs with shuffled data inside
    :param source_files_location:
        data set location
    :param destination_files_location:
        shuffled files catalogue
    :param num_to_class:
        dictionary of classes
    :param shuffled_csv_number:
        number of final CSVs with shuffled data
    :param rows_per_class:
        number of rows taken from source data file
    """
    # divide records among shuffled_csv_number files
    for i, category in tqdm(num_to_class.items()):
        df = read_train_simplified_csv(source_files_location, category, nrows=rows_per_class)
        df['hash'] = (df.key_id // 10 ** 7) % shuffled_csv_number  # let's say hash function
        for k in range(shuffled_csv_number):
            filename = destination_files_location+'train_{}.csv'.format(k)
            chunk = df[df.hash == k]
            chunk = chunk.drop(['key_id'], axis='columns')
            if i == 0:
                chunk.to_csv(filename, index=False)
            else:
                chunk.to_csv(filename, mode='a', header=False, index=False)

    # randomize records inside files
    for k in tqdm(range(shuffled_csv_number)):
        filename = destination_files_location+'train_{}.csv'.format(k)
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df['rnd'] = np.random.rand(len(df))
            df = df.sort_values(by='rnd').drop('rnd', axis='columns')
            df.to_csv(filename + '.gz', compression='gzip', index=False)
            os.remove(filename)
            print(df.shape)


if __name__ == '__main__':
    start = datetime.datetime.now()
    # Getting Data
    train_files = os.listdir(FILES_LOCATION + "train_simplified/")
    print(train_files)
    num_to_class = create_classes_dictionary(train_files)

    columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
    selected_categories = ['airplane', 'axe', 'book', 'bowtie', 'cake', 'calculator']

    train = None
    for i, category in enumerate(selected_categories) if run_selected_categories else num_to_class.items():
        print(category)
        if i == 0:
            train = read_train_simplified_csv(FILES_LOCATION, category, nb_training_cases_per_cat)
        else:
            train = train.append(read_train_simplified_csv(FILES_LOCATION, category, nb_training_cases_per_cat))
            # sub_train = sub_train[sub_train.recognized] # TODO lets think about this filtration
        print(train.shape)
        print(train.shape[1])

    print('!!! END !!! Train Test Size:', train.shape)
    print(train.head(5))

    # be careful when changing the number of files and the number of records from one file !!!
    # it takes time and disc space !!!
    #create_shuffled_data(FILES_LOCATION, FILES_LOCATION + "shuffled/", num_to_class, SHUFFLED_CSV_NUMBER, ROWS_PER_CLASS)

    # Data transformation
    train['input'] = None
    train['input'] = train.drawing.apply(lambda y: create_bitmap(str(y)))
    print(train.input[:5])

    train = train[['input', 'word']]
    train.word = train.word.astype('category')
    print(train.head(5))

    X_train, X_test, y_train, y_test = train_test_split(train.input, train.word)

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
    create_dummy_submission(FILES_LOCATION, num_to_class)

    end = datetime.datetime.now()
    print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

