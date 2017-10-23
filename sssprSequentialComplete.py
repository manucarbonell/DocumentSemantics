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

batch_size=1
max_non_improving_epochs=50
min_epochs=2
verbose_period=5
im_height=80
im_width=125
im_depth=1
max_seq_len=35
learning_rate=0.001
lr_decay=0.0001
max_epochs=400

visualize=True #Training accuracies visualization

experiment_id=os.path.splitext(__file__)[0]


def buildModel():
    inputimages = Input(shape=(None,im_height,im_width, im_depth))

    inputimage=Input(shape=(im_height,im_width,im_depth))

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

    x2=Bidirectional(LSTM(128,return_sequences=True,dropout=0.5,recurrent_dropout=0.5))(x2)

    category_output=TimeDistributed(Dense(6,activation='softmax'))(x2)
    person_output = TimeDistributed(Dense(8, activation='softmax'))(x2)
    optimizer=SGD(lr=learning_rate,momentum=0.9,nesterov=True,decay=lr_decay)
    m=Model(inputs=[inputimages],outputs=[category_output,person_output])
    m.compile(loss=['categorical_crossentropy','categorical_crossentropy'],optimizer=optimizer,metrics=['acc'])

    return m

def smoothlabel(x,amount=0.25,variance=5):
    mu=amount/x.size
    
    sigma=mu/variance
    noise=np.random.normal(mu,sigma,x.shape)
    smoothed=x*(1-noise.sum())+noise
    return smoothed

def visualize_training():
    cat_train_acc=[]
    cat_valid_acc=[]
    pers_train_acc=[]
    pers_valid_acc=[]

    log_file = open("./training_log/" + experiment_id + ".txt", 'r')
    header=log_file.readline()
    epochs=log_file.readlines()

    for epoch in epochs:
        accuracies=epoch.split("\t")
        cat_train_acc.append(float(accuracies[0]))
        cat_valid_acc.append(float(accuracies[1]))
        pers_train_acc.append(float(accuracies[2]))
        pers_valid_acc.append(float(accuracies[3]))
    import matplotlib.pyplot as plt
    plt.plot(cat_train_acc,color='r')
    plt.plot(cat_valid_acc, color='g')
    plt.title("Category accuracy, train red valid green")
    plt.show(block=False)

    plt.plot(pers_train_acc, color='r')
    plt.plot(pers_valid_acc, color='g')
    plt.title("Person accuracy, train red valid green")
    plt.show(block=False)

def trainModel(m):
    print "Train model..."

    E=EsposallesDataset(cvset='train')
    Ev=EsposallesDataset(cvset='validation')

    if not os.path.exists('saved_weights'):
        os.mkdir('saved_weights')

    log_file = open("./training_log/" + experiment_id + ".txt", 'w')
    log_file.write("Categ train acc\tCateg valid acc\tPerson train acc\tPerson valid acc\n")
    log_file.close()

    non_improving_epochs=0
    bestValidationACC=0

    for epoch in range(max_epochs):
        print 'Epoch: ',epoch,'================='
        train_categ_accs=[]
        train_pers_accs = []
        
        ####### TRAINING EPOCH #########

        for j in range(E.epoch_size/batch_size):
            word_images,categories,persons,ids=E.get_batch()
            categories = smoothlabel(categories)
            persons=smoothlabel(persons)

            total_loss, categ_loss, pers_loss, categ_acc, pers_acc = m.train_on_batch(word_images,y=[categories, persons])

            train_categ_accs.append(categ_acc)
            train_pers_accs.append(pers_acc)

        valid_categ_accs = []
        valid_pers_accs = []

        ###### VALIDATION EPOCH #######

        for j in range(Ev.epoch_size / batch_size):
            word_images, categories, persons, ids = Ev.get_batch()
            categories = smoothlabel(categories)
            persons = smoothlabel(persons)

            total_loss, categ_loss, pers_loss, categ_acc, pers_acc = m.evaluate(word_images,y=[categories, persons],verbose=0)
            valid_categ_accs.append(categ_acc)
            valid_pers_accs.append(pers_acc)
        log_file = open("./training_log/"+experiment_id+".txt", 'a')
        log_file.write(str(np.mean(train_categ_accs))+"\t"+str(np.mean(valid_categ_accs))+"\t"+str(np.mean(train_pers_accs))+"\t"+str(np.mean(valid_pers_accs))+"\n")
        log_file.close()

        ValidationACC=np.mean([np.mean(valid_pers_accs),np.mean(valid_categ_accs)])
        
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
    m.load_weights('./saved_weights/' + experiment_id + '_esposalles.h5')
    for j in xrange (E.epoch_size/batch_size):
        word_images,categories,persons,ids=E.get_batch();
        total_loss, categ_loss, pers_loss, categ_acc, pers_acc = m.evaluate(word_images, y=[categories, persons],verbose=0)
        categ_accs.append(categ_acc)
        categ_losses.append(categ_loss)
        pers_accs.append(pers_acc)
        pers_losses.append(pers_loss)
    print "TEST CATEGORY ACCURACY:",np.mean(categ_acc)
    print "TEST PERSON ACCURACY:",np.mean(pers_acc)


def generateTestCSV(m,outFilename='output.csv'):

    E=EsposallesDataset(cvset='test')
    m.load_weights('./saved_weights/' + experiment_id + '_esposalles.h5')

    with open(outFilename,mode='w') as outfile:
        for j in range(E.epoch_size/batch_size):
            word_images,categories,persons,ids=E.get_batch()
            pred_pers_cat=m.predict_on_batch([word_images])
            pred_categories=pred_pers_cat[0]
            pred_persons=pred_pers_cat[1]

            persons_categories_out=E.get_labels_from_categorical(ids,pred_categories,pred_persons)

            for record  in persons_categories_out[0]:
                word_id,category,person = record
                outfile.write("%s,%s,%s\n"%(word_id,category,person))

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
