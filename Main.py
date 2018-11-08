#Imports
from config import main_config as mc #user specific config
import pandas as pd
import numpy as np
import os
import datetime

#Imports for the CNN
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense




#Parameters
filelocation = mc.DATA_SET_DIR_PATH
nb_training_cases_per_cat = 100
RunForSelectedCategories = True


def create_dummy_submission(dataset_location, num_to_class):
    """ creates submission with random chosen classes assigned to test drawings """
    test_simplified_df = pd.read_csv(dataset_location+"test_simplified.csv")
    submission = test_simplified_df[['key_id']]
    ids = np.random.randint(0, len(num_to_class), 3)
    submission['word'] = " ".join([num_to_class[k] for k in ids if k in num_to_class])
    submission.to_csv('submissions/submission_{}.csv'.format(datetime.date.today()),
                      index=False,
                      #header=['key_id', 'word']
                      )


#Getting Data
train_files = os.listdir(filelocation+"train_simplified/")
num_to_class = {i: v[:-4].replace(" ", "_") for i, v in enumerate(train_files)} #adds underscores

columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
selected_categories=['airplane', 'axe', 'book', 'bowtie', 'cake', 'calculator']

if RunForSelectedCategories:
    train_files = [x+'.csv' for x in selected_categories]
    print(train_files)

train = pd.read_csv(filelocation + 'train_simplified/' + train_files[0], nrows=nb_training_cases_per_cat)
for file in train_files[1:]:
    print(file)
    sub_train=pd.read_csv(filelocation + 'train_simplified/' + file, nrows=nb_training_cases_per_cat)
    sub_train = sub_train[sub_train.recognized==True]
    train = train.append(sub_train)
    print(train.shape)
    print(train.shape[1])

print('!!! END !!! Train Test Size:', train.shape)
print(train.head(5))

### Filter for selected categories
# train_selected = train.loc[train['word'].isin(selected_categories)]

#Data transformation

#Model
#CNN
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), padding= 'same', input_shape = (64,64, 1), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(32, kernel_size = (3,3), padding = 'same', activation= 'relu'))
model.add(MaxPooling2D(pool_size= (2,2)))
model.add(Flatten())
model.add(Dense(train.shape[0],activation = 'softmax'))
model.summary()

#Compilation of the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])




#Model Evaluation

#Export results to submission file
create_dummy_submission(filelocation, num_to_class)

