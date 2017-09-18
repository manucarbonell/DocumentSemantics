from datasets.EsposallesCompetition import EsposallesDataset
#from layers.SpatialPyramidPooling import SPP
from kerasSPP.SpatialPyramidPooling import SpatialPyramidPooling
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
from keras.layers import Input
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM

from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K

import os
import numpy as np
import config
import time
experiment_id=os.path.splitext(__file__)[-1]+time.strftime("%Y%m%d_%H%M")
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

    outputs=TimeDistributed(Dense(6,activation='softmax'))(x2)
    optimizer=SGD(lr=config.learning_rate,momentum=0.9,nesterov=True,decay=config.lr_decay)
    m=Model(inputs=[inputimages],outputs=[outputs])
    m.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])

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

    for epoch in range(100):
        print 'Epoch: ',epoch,'================='
        accs=[]
        losses=[]

        for j in range(E.epoch_size/config.batch_size):

            x,y,example_id=E.get_batch()
            l,a=m.train_on_batch([x],y)#,class_weight=E.class_weights)
            accs.append(a)
            losses.append(l)

        print 'avg training loss:  ',np.mean(losses),'avg training accuracy:  ',np.mean(accs)
        accs=[]

        for jv in range(E.epoch_size/config.batch_size):
            x,y,example_id=Ev.get_batch()
            #y=smoothlabel(y)
            l,a=m.evaluate([x],y,verbose=0)
            accs.append(a)
            losses.append(l)
        print 'avg validation loss:',np.mean(losses),'avg validation accuracy:',np.mean(accs)
        ValidationACC=np.mean(accs)
        if ValidationACC>bestValidationACC:
            print 'new best validation accuracy', ValidationACC,'epoch:',epoch
            bestValidationACC=ValidationACC
            non_improving_epochs=0
            m.save_weights('./saved_weights/'+experiment_id+'_esposalles.h5',overwrite=True)
        else:
            non_improving_epochs+=1
            if non_improving_epochs>max_non_improving_epochs and epoch > min_epochs:
                print max_non_improving_epochs,' epochs without improving validation accuracy. Training Finished'
                break
    print 'done'

def evaluateModel(m,show_confmat=False):
    E=EsposallesDataset(cvset='test')
    m.load_weights('./saved_weights/'+experiment_id+'_esposalles.h5')
    revdict={v:k for (k,v) in E.labeldict.iteritems()}
    accs=[]
    losses=[]

    for j in xrange (E.epoch_size/config.batch_size):
        x,y,example_id=E.get_batch();

        l,a=m.evaluate([x],y,verbose=0)
        accs.append(a)
        losses.append(l)
    print np.mean(accs)

def main():
    m=buildModel()

    trainModel(m)
    print "Training finished."

    print "Testing model..."
    evaluateModel(m)


if __name__=="__main__":
    main()
