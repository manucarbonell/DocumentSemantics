from datasets.EsposallesCompetition import EsposallesDataset
#from layers.SpatialPyramidPooling import SPP
from kerasSPP.SpatialPyramidPooling import SpatialPyramidPooling
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Dense,Activation,Dropout
from keras.utils import np_utils
from keras.layers import Input
from keras.layers.merge import Concatenate
from keras.models import Model
import keras.backend as K

import os
import numpy as np
import config

experiment_id=os.path.splitext(__file__)[0]
max_non_improving_epochs=config.max_non_improving_epochs
min_epochs=config.min_epochs
verbose_period=config.verbose_period
def buildModel():
    if K.backend()=='tensorflow':
        inputimage=Input(shape=(None,None,1))
    else:
        inputimage=Input(shape=(1,None,None))

    inputlabel=Input(shape=(7,))
    x=Conv2D(32,(3,3),padding='same',activation='relu')(inputimage)
    x=Conv2D(32,(3,3),padding='same',activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x=Conv2D(64,(3,3),padding='same',activation='relu')(x)
    x=MaxPooling2D(pool_size=(2,2))(x)
    x=Conv2D(128,(3,3),padding='same',activation='relu')(x)
    x=Conv2D(128,(3,3),padding='same',activation='relu')(x)
    x=Conv2D(256,(3,3),padding='same',activation='relu')(x)
    if K.backend()=='tensorflow':
        x=SpatialPyramidPooling([1, 2, 4])(x)
    else:
        x=SPP([(4,4),(2,2),(1,1)])(x)
    y=Dense(64,activation='relu')(inputlabel)
    y=Dropout(0.5)(y)
    y=Dense(128,activation='relu')(y)
    y=Dropout(0.5)(y)
    y=Dense(128,activation='relu')(y)

    x=Concatenate()([x,y])
    x=Dense(2048,activation='relu')(x)
    x=Dropout(0.5)(x)
    x=Dense(512,activation='relu')(x)
    x=Dropout(0.5)(x)
    outputs=Dense(6,activation='softmax')(x)

    m = Model(inputs=[inputimage, inputlabel], outputs=[outputs])
    optimizer=SGD(lr=0.0001,momentum=0.9,nesterov=True,decay=0.000001)
    m.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['acc'])
    return m

def smoothlabel(x,amount=0.25,variance=5):
    mu=amount/x.size
    sigma=mu/variance
    noise=np.random.normal(mu,sigma,x.shape)
    smoothed=x*(1-noise.sum())+noise
    return smoothed

def noisylabel(x,prob=0.3):
    if np.random.random()<prob:
       return np.random.permutation(x)
    return x

def getPreviousLabel(example,previous_prediction,true_previous_label,dataset):
    new_register=np.asarray([[0.,0.,0.,0.,0.,0.,1.]],dtype='float32')
    previous_label=np_utils.to_categorical([dataset.prevlabels[example]],7)
    if true_previous_label or (previous_label==new_register).all():
        return previous_label
    if previous_prediction.size==7:
        return previous_prediction
    return np.hstack((previous_prediction==np.max(previous_prediction),[[0.]]))

def trainModel(m):
    print "Train model..."
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
        predicted_label=np.asarray([[0.,0.,0.,0.,0.,0.,1.]],dtype='float32') #Start register

        for j in xrange (E.numberSamples):
            x,y,example_id=E.get_example()

            y=np_utils.to_categorical(y,6)
            y=smoothlabel(y)
            previous_label=getPreviousLabel(example_id,predicted_label,False,E)
            predicted_label=m.predict_on_batch([x,previous_label])
            l,a=m.train_on_batch([x,previous_label],y)#,class_weight=E.class_weights)
            if j % verbose_period == 0 and j>0:
                print "Epoch",epoch,"step",j," loss ",np.sum(losses[j-verbose_period:j])
            if j % 500  == 0:
                m.save_weights('./saved_weights/'+experiment_id+'_esposalles.h5',overwrite=True)
            accs.append(a)
            losses.append(l)

        print 'avg training loss:  ',np.mean(losses),'avg training accuracy:  ',np.mean(accs)
        accs=[]
        predicted_label=np.asarray([[0.,0.,0.,0.,0.,0.,1.]],dtype='float32') #Start register
        for jv in xrange (Ev.numberSamples):
            x,y,example_id=Ev.get_example()
            y=np_utils.to_categorical(y,6)
            y=smoothlabel(y)
            previous_label=getPreviousLabel(example_id,predicted_label,False,Ev)
            predicted_label=m.predict_on_batch([x,previous_label])
            l,a=m.evaluate([x,previous_label],y,verbose=0)#,class_weight=E.class_weights)
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

def evaluateModel(m,true_previous_label=False,show_confmat=False):
    E=EsposallesDataset(cvset='test')
    m.load_weights('./saved_weights/'+experiment_id+'_esposalles.h5')
    revdict={v:k for (k,v) in E.labeldict.iteritems()}
    accs=[]
    losses=[]
    confmat=np.zeros((6,6),dtype='int32')
    predicted_label=np.asarray([[0.,0.,0.,0.,0.,0.,1.]],dtype='float32') #Start register
    print E.numberSamples
    for j in xrange (E.numberSamples):
        x,y,example_id=E.get_example();
        previous_label=getPreviousLabel(example_id,predicted_label,true_previous_label,E)
        #predict current label
        predicted_label=m.predict_on_batch([x,previous_label])
        y_pred=np.argmax(predicted_label[0])
        print "%s,%s"% (example_id,revdict[y_pred])#,revdict[y[0]]
        confmat[np.argmax(predicted_label),y[0]]+=1
        confmat[y_pred,y[0]]+=1
        y=np_utils.to_categorical(y,6)
        l,a=m.evaluate([x,previous_label],y,verbose=0)
        accs.append(a)
        losses.append(l)
    if true_previous_label:
       print 'best case',
    print 'accuracy,loss:',np.mean(accs),np.mean(losses)
    if show_confmat:
       print 'confmat:'
       print confmat



m=buildModel()
#print "Training..."
trainModel(m)
print "Testing..."
#evaluateModel(m,true_previous_label=True)
evaluateModel(m)
