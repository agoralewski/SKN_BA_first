#Imports
from config import main_config as mc #user specific config
import pandas as pd

#Parameters
filelocation = mc.DATA_SET_DIR_PATH

#Getting Data
train = pd.read_csv(filelocation+'train_simplified/tiger.csv')
print(train.shape)
print(train.head(5))

#Data transformation

#Model

#Model Evaluation

#Export results to submission file
#submission = test[['key_id', 'word']]
#submission.to_csv('first_submission.csv', index=False)
#submission.head()
