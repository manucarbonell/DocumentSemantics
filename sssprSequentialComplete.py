from datasets.EsposallesCompetition import EsposallesDataset

from kerasSPP.SpatialPyramidPooling import SpatialPyramidPooling
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Dense,Activation,Dropout
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM
from keras.models import Model
import matplotlib.pyplot as plt

import glob
import os
import sys
import numpy as np
import config
import time
experiment_id=sys.argv[0].split('.')[0]+time.strftime("%Y%m%d_%H%M")
max_non_improving_epochs=config.max_non_improving_epochs
min_epochs=config.min_epochs
verbose_period=config.verbose_period

def buildModel():
    inputimages = Input(shape=(None,config.im_height,config.im_width, config.im_depth))

    inputimage=Input(shape=(config.im_height,config.im_width,config.im_depth))

    x=Conv2D(32,(3,3),padding='same',activation='relu')(inputimage)
    x=Conv2D(32,(3,3),padding='same',activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Conv2D(128,(3,3),padding='same',activation='relu')(x)
    x=Conv2D(128,(3,3),padding='same',activation='relu')(x)
    x=Conv2D(256,(3,3),padding='same',activation='relu')(x)
    x=SpatialPyramidPooling([1, 2, 4])(x)
    x=Dense(2048,activation='relu')(x)
    x=Dropout(0.5)(x)
    out1=Dense(512,activation='relu')(x)
    sspr_model = Model(inputs=[inputimage], outputs=[out1])

    x2=TimeDistributed(sspr_model)(inputimages)

    x2=Bidirectional(LSTM(100,return_sequences=True,dropout=0.5,recurrent_dropout=0.5))(x2)

    category_output=TimeDistributed(Dense(6,activation='softmax'))(x2)
    person_output = TimeDistributed(Dense(8, activation='softmax'))(x2)
    optimizer=SGD(lr=config.learning_rate,momentum=0.9,nesterov=True,decay=config.lr_decay)
    m=Model(inputs=[inputimages],outputs=[category_output,person_output])
    m.compile(loss=['categorical_crossentropy','categorical_crossentropy'],optimizer=optimizer,metrics=['acc'])

    return m

def smoothlabel(x,amount=0.25,variance=5):
    mu=amount/x.size
    sigma=mu/variance
    noise=np.random.normal(mu,sigma,x.shape)
    smoothed=x*(1-noise.sum())+noise
    return smoothed


def trainModel(m):
    print "Train model..."
    print "Training parameters:"
    os.system("cat config.py")
    E=EsposallesDataset(cvset='train')
    Ev=EsposallesDataset(cvset='validation')

    if not os.path.exists('saved_weights'):
        os.mkdir('saved_weights')

    non_improving_epochs=0
    bestValidationACC=0

    for epoch in range(config.max_epochs):
        print 'Epoch: ',epoch,'================='
        accs=[]
        losses=[]

        for j in range(E.epoch_size/config.batch_size):

            word_images,categories,persons,ids=E.get_batch()
            categories = smoothlabel(categories)
            persons=smoothlabel(persons)
            y=np.concatenate((persons,categories),axis=2)
            l,a=m.train_on_batch(word_images,y)

            accs.append(a)
            losses.append(l)

        print 'avg training loss:  ',np.mean(losses),'avg training accuracy:  ',np.mean(accs)
        accs=[]

        for jv in range(E.epoch_size/config.batch_size):
            word_images,categories,persons,ids=Ev.get_batch()
            categories=smoothlabel(categories)
            l,a=m.evaluate([word_images],categories,verbose=0)
            accs.append(a)
            losses.append(l)
        print 'avg validation loss:',np.mean(losses),'avg validation accuracy:',np.mean(accs)
        ValidationACC=np.mean(accs)
        if ValidationACC>bestValidationACC:
            print 'New best validation accuracy', ValidationACC,'epoch:',epoch
            bestValidationACC=ValidationACC
            non_improving_epochs=0
            m.save_weights('./saved_weights/'+experiment_id+'_esposalles.h5',overwrite=True)
        else:
            non_improving_epochs+=1
            if non_improving_epochs>max_non_improving_epochs and epoch > min_epochs:
                print max_non_improving_epochs,' epochs without improving validation accuracy. Training Finished'
                break
    print 'done'

def load_latest_model(m):
    list_of_models = glob.glob('./saved_weights/*.h5')
    if len(list_of_models) > 0:
        latest_model = max(list_of_models, key=os.path.getctime)

        m.load_weights(latest_model)
    else:
        print "NO MODEL TO EVALUATE. Use", sys.argv[0], 'train'
        sys.exit()

    return m


def evaluateModel(m):
    E=EsposallesDataset(cvset='test')
    accs=[]
    losses=[]
    m=load_latest_model(m)
    for j in xrange (E.epoch_size/config.batch_size):
        word_images,categories,persons,ids=E.get_batch();
        l,a=m.evaluate([word_images],categories,verbose=0)
        accs.append(a)
        losses.append(l)
    print "TEST ACCURACY:",np.mean(accs)


def generateTestCSV(m,outFilename='output.csv'):
    E=EsposallesDataset(cvset='test')
    m=load_latest_model(m)
    with open(outFilename,mode='w') as outfile:
        for j in range(E.epoch_size/config.batch_size):
            word_images,categories,persons,ids=E.get_batch()
            categories_pred=m.predict_on_batch([word_images])
            out=E.get_labels_from_categorical(ids,categories_pred)
            for record  in out:
                for word_id,category in record:
                    outfile.write("%s,%s\n"%(word_id,category))


def main():

    if len(sys.argv)>1:
        mode=sys.argv[1]
    else:
        mode=None
    if mode=='train':
        m = buildModel()
        trainModel(m)
        print "Training finished."
    elif mode=='eval':
        m = buildModel()
        print "Testing model..."
        evaluateModel(m)

    elif mode=='csvout':
        m = buildModel()
        generateTestCSV(m)

    else:
        print "Usage: python",sys.argv[0],'mode=[train,eval,traineval,csvout]'

if __name__=="__main__":
    main()
