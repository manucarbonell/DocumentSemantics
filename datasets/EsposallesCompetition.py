# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:12:24 2016

@author: jtoledo
"""

import numpy as np
import re
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
import sys
import os
from PIL import Image
import matplotlib.pyplot as plt
from keras.utils import np_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

batch_size=config.batch_size
im_height=config.im_height
im_width=config.im_width

class EsposallesDataset():
    def __init__(self,BaseDir='/home/ntoledo/datasets/OfficialEsposalles',cvset='train',level='word'):
        self.BaseDir=BaseDir
        self.DataDir=BaseDir+'/'+cvset
        self.GroundTruth=self.DataDir+'/category_groundtruth.txt'
        self.level=level
        self.labels={}
        with open(self.GroundTruth, mode='r') as infile:
            for line in infile:
                values=line.rstrip('\n').split(':',2)
                self.labels[values[0]]=values[1]
        self.labeldict={l:n for (n,l) in enumerate(['other','surname','name','location','occupation','state'])}
        self.generate_previous_labels_and_regdict()
        self.shuffled_registers=[k for k in self.reg_dict.iterkeys()]
        self.register_iterator=iter(self.shuffled_registers)
        self.epoch_size = len(self.shuffled_registers)
        self.r_id=self.register_iterator.next()
        self.word_iterator=iter(self.reg_dict[self.r_id])
        self.w_id=-1
        self.epoch=0
        self.numberSamples=len(self.labels.keys())

    def readNormalizedImage(self,imageid):
        # Read image from file
        v=imageid.split('_');
        filename=self.DataDir+'/'+'_'.join(v[0:2])+'/words/'+imageid+".png"
        im=Image.open(filename)

        #Resize image to fit maximum size
        maxsize = (config.im_height,config.im_width)
        im.thumbnail(maxsize)

        #Create background to paste image into
        background=np.zeros(maxsize)
        im=np.array(im)
        #average_bckg_color=int(np.mean(im[np.where(im<100)]))
        average_bckg_color = int(np.mean(im))
        background.fill(average_bckg_color)
        background=Image.fromarray(background)

        #Paste image centered into grey background
        im=Image.fromarray(im)
        background.paste(im,box=(maxsize[1]/2-im.size[0]/2,maxsize[0]/2-im.size[1]/2))
        im=np.array(background)
        im= 1.-np.array(im)/255.

        return im


    def generate_previous_labels_and_regdict(self):
        l=self.labels.keys()
        l=sorted(l,key=natural_key)
        prevlabel=6
        prevpag,prevreg,pag,reg=(0,0,0,0)
        self.prevlabels={}
        self.reg_dict={}
        for s in l:
            m=re.search('^idPage([0-9]{5})_Record([0-9]{1,2})_Line([0-9]{1,2})_Word([0-9]{1,2})$',s);
            pag,reg,lin,pos=int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4))
            if self.reg_dict.has_key((pag,reg)):
               self.reg_dict[pag,reg].append(s)
            else:
               self.reg_dict[pag,reg]=[s]
            if prevpag==0 or prevpag==pag and prevreg == reg:
               self.prevlabels[s]=prevlabel
               prevlabel=self.labeldict[self.labels[s]]
            else:
               prevlabel=6
               self.prevlabels[s]=prevlabel
            prevpag=pag
            prevreg=reg

    def get_batch(self,show_word_ids=False):

        self.word_iterator=iter(self.reg_dict[self.r_id])
        IDS=[i for i in self.word_iterator]

        self.word_iterator = iter(self.reg_dict[self.r_id])
        X=[np.asarray(self.readNormalizedImage(i))[:,:,np.newaxis] for i in self.word_iterator]

        self.word_iterator=iter(self.reg_dict[self.r_id])
        Y=[np.asarray(self.labeldict[self.labels[i]]) for i in self.word_iterator]

        try:
            self.r_id=self.register_iterator.next()
        except StopIteration: #if no more register, shuffle and get first register again
            np.random.shuffle(self.shuffled_registers)
            self.register_iterator=iter(self.shuffled_registers)
            self.r_id=self.register_iterator.next()
            self.epoch+=1

        X = np.stack(X)
        X=X[np.newaxis,:,:,:,:]
        Y = np_utils.to_categorical(Y, config.n_classes)
        Y=Y[np.newaxis,:,:]

        return X,Y,IDS


    def get_transcription_from_categorical(self,predictions):
        if self.revdict is None: self.revdict={v:k for (k,v) in self.labeldict.iteritems()}
        return [','.join([self.revdict[timestep] for timestep in sample]).rstrip(',0') for sample in predictions]

    def show_batch(self):

        ims,labs,names=self.get_batch()
        widths=[]
        heights=[]

        for i in range(len(ims[0,:,:,:,0])):
            im=ims[0,i,:,:,0]

            print im.shape

            im=im.astype('uint8')
            im=Image.fromarray(im)
            im.show()
            plt.imshow(im)


            print names[i],labs[0,i,:]
            plt.show()

def main():
    E=EsposallesDataset()
    E.show_batch()

if __name__=="__main__":
    main()