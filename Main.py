#Imports
from config import main_config as mc #user specific config
import pandas as pd
import os

#Parameters
filelocation = mc.DATA_SET_DIR_PATH
nb_training_cases_per_cat = 100

#Getting Data
train_files = os.listdir(filelocation+"train_simplified/")
print(train_files[:5])
columns = ['countrycode', 'drawing', 'key_id', 'recognized', 'timestamp', 'word']
selected_categories=['airplane', 'axe' 'book' 'bowtie', 'cake' 'calculator']

train = pd.read_csv(filelocation + 'train_simplified/' + train_files[0]).head(nb_training_cases_per_cat)
for file in train_files[1:61]:
    sub_train=pd.read_csv(filelocation + 'train_simplified/' + file).head(nb_training_cases_per_cat)
    train = train.append(sub_train)
    print(train.shape)

print('!!! END !!! Train Test Size:', train.shape)
print(train.head(5))

### Filter for selected categories
train_selected = train.loc[train['word'].isin(selected_categories)]

#Data transformation

#Model

#Model Evaluation

#Export results to submission file
#submission = test[['key_id', 'word']]
#submission.to_csv('first_submission.csv', index=False)
#submission.head()
