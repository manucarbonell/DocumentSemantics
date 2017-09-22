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

def load_latest_model(m):
    list_of_models = glob.glob('./saved_weights/*.h5')
    if len(list_of_models) > 0:
        latest_model = max(list_of_models, key=os.path.getctime)
        if latest_model.split("_")==sys.argv[0].split(".")[0]:
            m.load_weights(latest_model)
            print "MODEL RESTORED."
        print "NO MODEL TO LOAD."
    else:
        print "NO MODEL TO LOAD."
        log_file = open("./training_log/"+experiment_id+".txt", 'w')
        log_file.write("Categ train acc\tCateg valid acc\tPerson train acc\tPerson valid acc\n")
        log_file.close()
    return m




def trainModel(m):
    load_latest_model(m)    
    print "Train model..."
    print "Training parameters:"
    os.system("cat config.py")
    print "\n"
    E=EsposallesDataset(cvset='train')
    Ev=EsposallesDataset(cvset='validation')

    if not os.path.exists('saved_weights'):
        os.mkdir('saved_weights')

    non_improving_epochs=0
    bestValidationACC=0

    #for epoch in range(config.max_epochs):
    for epoch in range(10):
        print 'Epoch: ',epoch,'================='
        train_categ_accs=[]
        train_pers_accs = []
        
        ####### TRAINING EPOCH #########

        for j in range(E.epoch_size/config.batch_size):
            word_images,categories,persons,ids=E.get_batch()
            categories = smoothlabel(categories)
            persons=smoothlabel(persons)

            total_loss, categ_loss, pers_loss, categ_acc, pers_acc = m.train_on_batch(word_images,y=[categories, persons])

            train_categ_accs.append(categ_acc)
            train_pers_accs.append(pers_acc)

        valid_categ_accs = []
        valid_pers_accs = []

        ###### VALIDATION EPOCH #######

        for j in range(Ev.epoch_size / config.batch_size):
            word_images, categories, persons, ids = Ev.get_batch()
            categories = smoothlabel(categories)
            persons = smoothlabel(persons)

            total_loss, categ_loss, pers_loss, categ_acc, pers_acc = m.evaluate(word_images,y=[categories, persons],verbose=0)
            valid_categ_accs.append(categ_acc)
            valid_pers_accs.append(pers_acc)
        log_file = open("./training_log/"+experiment_id+".txt", 'a')
        log_file.write(str(np.mean(train_categ_accs))+"\t"+str(np.mean(valid_categ_accs))+"\t"+str(np.mean(train_pers_accs))+"\t"+str(np.mean(valid_pers_accs))+"\n")
        log_file.close()
        
        ValidationAcc=np.mean([np.mean(valid_pers_accs),np.mean(valid_categ_accs)])
        
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


def evaluateModel(m):
    E=EsposallesDataset(cvset='test')
    categ_accs = []
    pers_accs = []
    categ_losses = []
    pers_losses = []
    m=load_latest_model(m)
    for j in xrange (E.epoch_size/config.batch_size):
        word_images,categories,persons,ids=E.get_batch();
        total_loss, categ_loss, pers_loss, categ_acc, pers_acc = m.evaluate(word_images, y=[categories, persons],verbose=0)
        categ_accs.append(categ_acc)
        categ_losses.append(categ_loss)
        pers_accs.append(pers_acc)
        pers_losses.append(pers_loss)
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
