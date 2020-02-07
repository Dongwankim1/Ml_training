import numpy as np
import pandas as pd
import tensorflow as tf
import sklearn
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('E:/dacondata/천체/train.csv',index_col='id')
unique_labels = train['type'].unique()
label_dict = {val : i for i, val in enumerate(unique_labels)}
i2lb = {v:k for k,v in label_dict.items()}
scaler = StandardScaler()
labels = train['type']
train =train.drop(columns=['type'])
_mat = scaler.fit_transform(train)
train = pd.DataFrame(_mat,columns=train.columns,index=train.index)
train_x = train.values
train_y = labels.replace(label_dict)



from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, BatchNormalization
from keras.layers import Bidirectional, GlobalMaxPool1D, MaxPooling1D, Add, Flatten
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.layers import BatchNormalization
from keras import regularizers
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from keras.layers import Activation
from keras.utils.generic_utils import get_custom_objects
from keras.models import Model, load_model
import time


acciracu = []

class SaveDataSub():
    def __init__(self,path=""):
        super(SaveDataSub,self).__init__()
        self.path = path


    def saveFile(self):

        test = pd.read_csv('E:/dacondata/천체/test.csv').reset_index(drop=True)
        test_ids = test['id']
        test = test.drop(columns=['id'])
        test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

        pred_mat = model.predict(test)

        sample = pd.read_csv('E:/dacondata/천체/sample_submission.csv')

        submission = pd.DataFrame(pred_mat, index=test.index)
        submission = submission.rename(columns=i2lb)
        submission = pd.concat([test_ids, submission], axis=1)
        submission = submission[sample.columns]
        submission.to_csv("E:/dacondata/천체/submission1.csv", index=False)


class ActivationCategory(Activation):
    def __init__(self, activation, **kwargs):
        super(ActivationCategory, self).__init__(activation, **kwargs)
        self.__name__ = 'gelu'



def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

get_custom_objects().update({'gelu': ActivationCategory(gelu)})




class ModelCategory():
    def __init__(self,activation_fuc,input_dim=0,dr_rate=0.05):
        super(ModelCategory,self).__init__()
        self.input_dim = input_dim
        self.activation_fuc = activation_fuc
        self.dr_rate = dr_rate

    def Sparse_categorical_crossentropy(self):

        inp = Input(shape = (self.input_dim,))
        x = Dense(256,activation=None)(inp)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(224, activation=None)(x)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(224, activation=None)(x)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(192, activation=None)(x)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(160, activation=None)(x)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(128, activation=None)(x)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(96, activation=None)(x)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(20, activation=None)(x)
        x = Activation(self.activation_fuc)(x)
        x = Dropout(self.dr_rate)(x)
        x = Dense(19, activation="softmax",name='type')(x)
        model = Model(inputs=inp,output=x)
        return model

    def Random_froest(self):

        pass


if __name__ == "__main__":
    mode =0
    date = round(time.time())
    mc = ModelCheckpoint('E:/dacondata/천체/model//model_.hdf5'.format(str(date)), monitor='val_acc', verbose=1, save_best_only=True)
    es = EarlyStopping(monitor='val_acc', patience=150)

    model=ModelCategory(gelu,input_dim=21).Sparse_categorical_crossentropy()
    model.summary()



    acciracu = []
    if(mode ==1):
        k_fold=30
        train_x = train_x.astype(float)

        skf = KFold(n_splits=k_fold, shuffle=True)
        for enum, (train, validation) in enumerate(skf.split(train_x, train_y)):
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

            model.fit(train_x[train], train_y[train], validation_data=(train_x[validation], train_y[validation]), epochs=300,
                      batch_size=300, callbacks=[es, mc])



    if(mode==0):
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])

        model.fit(train_x, train_y, validation_split=0.05, epochs=300,batch_size=300, callbacks=[es, mc])

        test = pd.read_csv('E:/dacondata/천체/test.csv').reset_index(drop=True)
        test_ids = test['id']
        test = test.drop(columns=['id'])
        test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)

        pred_mat = model.predict(test)

        sample = pd.read_csv('E:/dacondata/천체/sample_submission.csv')

        submission = pd.DataFrame(pred_mat, index=test.index)
        submission = submission.rename(columns=i2lb)
        submission = pd.concat([test_ids, submission], axis=1)
        submission = submission[sample.columns]
        submission.to_csv("E:/dacondata/천체/submission1.csv", index=False)

















#
#
# for enum, (train, validation) in enumerate(skf.split(train_x, train_y)):
#     drop_lr = 0.05
#
#     mc = ModelCheckpoint('model/model_k_fold_{}.hdf5'.format(enum), monitor='val_acc', verbose=1, save_best_only=True)
#     es = EarlyStopping(monitor='val_acc', patience=50)
#     model = Sequential()
#     model.add(Dense(units=256, activation='relu', input_dim=21))
#     model.add(Dropout(drop_lr))
#     model.add(Dense(units=224, activation='relu'))
#     model.add(Dropout(drop_lr))
#     model.add(Dense(units=224, activation='relu'))
#     model.add(Dropout(drop_lr))
#     model.add(Dense(units=192, activation='relu'))
#     model.add(Dropout(drop_lr))
#     model.add(Dense(units=160, activation='relu'))
#     model.add(Dropout(drop_lr))
#     model.add(Dense(units=128, activation='relu'))
#     model.add(Dropout(drop_lr))
#     model.add(Dense(units=96, activation='relu'))
#     model.add(Dense(units=20, activation='relu'))
#     model.add(Dense(units=19, activation='softmax'))
#     # 모델을 컴파일합니다.
#     model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
#
#     model.fit(train_x[train], train_y[train], validation_data=(train_x[validation], train_y[validation]), epochs=300,
#               batch_size=300, callbacks=[es, mc])
#
#
#
# from keras.models import load_model
#
# test = pd.read_csv('E:/dacondata/천체/test.csv').reset_index(drop=True)
# test_ids = test['id']
# test = test.drop(columns=['id'])
# test = pd.DataFrame(scaler.transform(test), columns=test.columns, index=test.index)
#
# k_fold = 19
#
# pred_test = np.zeros((len(test), 19))
#
# print(pred_test.shape)
#
# for i in range(k_fold):
#     print(i)
#     model = load_model("model/model_k_fold_{}.hdf5".format(i))
#     pred = np.array(model.predict(test))
#
#     pred_test += pred
#
# pred_mat = pred_test / k_fold
#
# sample = pd.read_csv('E:/dacondata/천체/sample_submission.csv', index_col=0)
# submission = pd.DataFrame(pred_mat, index=test.index)
# submission = submission.rename(columns=i2lb)
# submission = pd.concat([test_ids, submission], axis=1)
# submission = submission[sample.columns]
# submission.to_csv('E:/dacondata/천체/submission.csv')



