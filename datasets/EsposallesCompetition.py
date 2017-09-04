# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 15:12:24 2016

@author: jtoledo
"""
import csv
import numpy as np
import string
import cv2
import re
from keras import backend as K
from PIL import Image

class EsposallesDataset():
    def __init__(self,BaseDir='/home/ntoledo/datasets/OfficialEsposalles',cvset='train'):
        self.BaseDir=BaseDir
        self.DataDir=BaseDir+'/'+cvset
        self.GroundTruth=self.DataDir+'/category_groundtruth.txt'
        self.labels={}
        with open(self.GroundTruth, mode='r') as infile:
            for line in infile:
                values=line.rstrip('\n').split(':',2)
                self.labels[values[0]]=values[1]
        self.labeldict={l:n for (n,l) in enumerate(['other','surname','name','location','occupation','state'])}
        self.generate_previous_labels_and_regdict()
        self.shuffled_registers=[k for k in self.reg_dict.iterkeys()]
        self.register_iterator=iter(self.shuffled_registers)
        self.r_id=self.register_iterator.next()
        self.word_iterator=iter(self.reg_dict[self.r_id])
        self.w_id=-1
        self.epoch=0
        self.numberSamples=len(self.labels.keys())
    def generate_previous_labels_and_regdict(self):
        l=self.labels.keys()
        l.sort()
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
    def get_example(self,sequence=False):
        #minx=30
        #miny=30
        def readNormalizedImage(imageid):
            v=imageid.split('_');
            filename=self.DataDir+'/'+'_'.join(v[0:2])+'/words/'+imageid+".png"

            img=1.-cv2.imread(filename,cv2.IMREAD_GRAYSCALE)/255.
            height,width=np.shape(img)
            return np.pad(img,(max(0,30-height),max(0,30-width)),'constant')
            #return img
        #while True:
        if not sequence:
            try:
                 current_example=self.word_iterator.next()
                 self.w_id+=1
            except StopIteration: #If no more words, go next register
                   try:
                       self.r_id=self.register_iterator.next()
                   except StopIteration: #if no more register, shuffle and get first register again
                          np.random.shuffle(self.shuffled_registers)
                          self.register_iterator=iter(self.shuffled_registers)
                          self.r_id=self.register_iterator.next()
                          self.epoch+=1
                   self.word_iterator=iter(self.reg_dict[self.r_id])
                   current_example=self.word_iterator.next()
                   self.w_id=0
            X=readNormalizedImage(current_example)
            #if X.shape[-2] >= miny and X.shape[-1] >= minx:
            #   #print current_example,self.labels[current_example],X.shape[-2],X.shape[-1]
            #   break;
            #print 'Image too small: ',current_example,self.labels[current_example],X.shape[-2],X.shape[-1]
            Y=[self.labeldict[self.labels[current_example]]]
            xout=np.asarray(X)[np.newaxis,np.newaxis,:,:]
            if K.backend()=='tensorflow':
                xout=xout[0,:,:,:,np.newaxis]
            return xout,np.asarray(Y),current_example

        else:
            self.r_id=self.register_iterator.next()
            print self.reg_dict[self.r_id]
            self.word_iterator=iter(self.reg_dict[self.r_id])
            example_seq=[]
            labels_seq=[]
            words_seq=[]
            for word in self.word_iterator:
                X=readNormalizedImage(word)
                #if X.shape[-2] >= miny and X.shape[-1] >= minx:
                #   #print current_example,self.labels[current_example],X.shape[-2],X.shape[-1]
                #   break;
                #print 'Image too small: ',current_example,self.labels[current_example],X.shape[-2],X.shape[-1]
                Y=[self.labeldict[self.labels[word]]]
                xout=np.asarray(X)[np.newaxis,np.newaxis,:,:]
                if K.backend()=='tensorflow':
                    xout=xout[0,:,:,:,np.newaxis]
                example_seq.append(xout)
                labels_seq.append(Y)
                words_seq.append(word)
            return example_seq,labels_seq,words_seq

    def get_transcription_from_categorical(self,predictions):
        if self.revdict is None: self.revdict={v:k for (k,v) in self.labeldict.iteritems()}
        return [','.join([self.revdict[timestep] for timestep in sample]).rstrip(',0') for sample in predictions]



def test_input():
    ims,labs,names=EsposallesDataset(cvset='train').get_example(sequence=True)
    print EsposallesDataset(cvset='train').labeldict
    for i in range(len(ims)):
        im=ims[i]
        im=im*255
        im=im.astype('uint8')
        im=im[0][:,:,0]
        print labs[i],names[i]

        im=Image.fromarray(im)
        im.show()
        raw_input()
    '''im,lab,name=EsposallesDataset(cvset='train').get_example()
    im=im*255
    im=im.astype('uint8')
    im=im[0][:,:,0]
    print im,type(im),im.shape
    im=Image.fromarray(im)
    im.show()
    '''

#test_input()
