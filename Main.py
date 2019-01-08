# Imports
from builtins import print

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
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
# check what is the difference Conv2D vs Convolution2D
from keras.applications.mobilenet import preprocess_input


# Parameters
FILES_LOCATION = mc.DATA_SET_DIR_PATH
RUN_SELECTED_CATEGORIES = False  # True
SHUFFLED_CSV_NUMBER = 100
ROWS_PER_CLASS = 300

BITMAP_SIZE = 64
BATCH_SIZE = 400


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


def create_bitmap(input_json, bitmap_height=64, bitmap_width=64):
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


def create_categories_dictionary(train_files_names):
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


# TODO lets think about filtration of unrecognized drawings
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
    for class_id, category in tqdm(num_to_class.items()):
        df = read_train_simplified_csv(source_files_location, category, nrows=rows_per_class)
        df['class_id'] = class_id
        df['hash'] = (df.key_id // 10 ** 7) % shuffled_csv_number  # let's say hash function
        for k in range(shuffled_csv_number):
            filename = destination_files_location+'train_{}.csv'.format(k)
            chunk = df[df.hash == k]
            chunk = chunk.drop(['key_id'], axis='columns')
            if class_id == 0:
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


def print_basic_info(input_object):
    """ for debugging print basic info """
    print('--------input info------------')
    print('type: ', type(input_object))
    print('shape: ', input_object.shape)
    if isinstance(input_object, pd.core.frame.DataFrame):
        print('cols names: ', list(input_object))
        print('head of dataframe:\n', input_object.head(5))
    print('----------------------------------')


#  TODO
def data_generator(files_location, batch_size, csv_number, bitmap_size=64, num_classes=340):
    while True:
        for k in np.random.permutation(csv_number):
            filename = os.path.join(files_location, 'train_{}.csv.gz'.format(k))
            for df in pd.read_csv(filename, chunksize=batch_size):
                # df['drawing'] = df['drawing'].apply(json.loads)
                x = np.zeros((len(df), bitmap_size, bitmap_size, 1))
                for i, drawing_json in enumerate(df.drawing.values):
                    x[i, :, :, 0] = create_bitmap(drawing_json)
                x = preprocess_input(x).astype(np.float32)
                y = keras.utils.to_categorical(df.class_id, num_classes=num_classes)
                yield x, y


def predictions_to_category_ids(predictions):
    """
    it takes predictions and returns the three most probable categories

    :param predictions: Numpy array(s) of predictions.
    :return: DataFrame with cols ['a', 'b', 'c'] - "We're the three best friends that anybody could have" :)
    """
    return pd.DataFrame(np.argsort(-predictions, axis=1)[:, :3], columns=['a', 'b', 'c'])


if __name__ == '__main__':
    start = datetime.datetime.now()

    # Getting Data
    train_files = os.listdir(FILES_LOCATION + "train_simplified/")
    num_to_category = create_categories_dictionary(train_files)
    category_to_num = {val: key for key, val in num_to_category.items()}
    columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
    selected_categories = ['airplane', 'axe', 'book', 'bowtie', 'cake', 'calculator']

    # be careful when changing the number of files and the number of records from one file !!!
    # it takes time and disc space !!!
    #create_shuffled_data(FILES_LOCATION, FILES_LOCATION + "shuffled/", num_to_class, SHUFFLED_CSV_NUMBER, ROWS_PER_CLASS)

    train = None
    for i, category in enumerate(selected_categories) if RUN_SELECTED_CATEGORIES else num_to_category.items():
        if i == 0:
            train = read_train_simplified_csv(FILES_LOCATION, category, ROWS_PER_CLASS)
        else:
            train = train.append(read_train_simplified_csv(FILES_LOCATION, category, ROWS_PER_CLASS))
            # sub_train = sub_train[sub_train.recognized] # TODO lets think about this filtration
    print_basic_info(train)

    # Data transformation
    train['input'] = train.drawing.apply(lambda y: create_bitmap(str(y)))
    train['word'] = train.word.apply(lambda y: category_to_num[y.replace(" ", "_")])
    train = train[['input', 'word']]
    print_basic_info(train)

    shaped_data = np.zeros((len(train), BITMAP_SIZE, BITMAP_SIZE, 1))
    for i, bitmap in enumerate(train.input):
        shaped_data[i, :, :, 0] = bitmap
    print_basic_info(shaped_data)

    # {0, 1, .., n} -> {[1,0,0..0], [0,1,0..0], ..}
    y_data = keras.utils.to_categorical(train.word, num_classes=340)
    print_basic_info(y_data)

    x_train, x_test, y_train, y_test = train_test_split(shaped_data, y_data)
    print_basic_info(x_train)
    print_basic_info(x_test)
    print_basic_info(y_train)
    print_basic_info(y_test)

    # CNN
    classifier = Sequential()
    classifier.add(Conv2D(32, kernel_size=(3, 3), padding='same', input_shape=(64, 64, 1), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(340, activation='softmax')) #'relu', 'sigmoid'
    classifier.summary()
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # loss='binary_crossentropy'

    # Fitting the CNN
    classifier.fit(x_train, y_train, verbose=1, validation_data=(x_test, y_test))

    # Model Evaluation

    # Prediction
    test_simplified_df = pd.read_csv(FILES_LOCATION + "test_simplified.csv")
    submission = test_simplified_df[['key_id']]

    shaped_data = np.zeros((len(test_simplified_df), BITMAP_SIZE, BITMAP_SIZE, 1))
    for i, bitmap in enumerate(test_simplified_df.drawing.apply(lambda y: create_bitmap(str(y)))):
        shaped_data[i, :, :, 0] = bitmap

    test_predictions = classifier.predict(shaped_data, batch_size=160, verbose=1)

    top3ids = predictions_to_category_ids(test_predictions)
    print_basic_info(top3ids)

    # translate ids to categories names joined by space
    submission['word'] = top3ids.replace(num_to_category).apply(lambda a: ' '.join(a), axis=1)

    # Export results to submission file
    # create_dummy_submission(FILES_LOCATION, num_to_category)
    submission.to_csv('submissions/submission_{}.csv'.format(datetime.date.today()), index=False)

    end = datetime.datetime.now()
    print('Latest run {}.\nTotal time {}'.format(end, (end - start)))
