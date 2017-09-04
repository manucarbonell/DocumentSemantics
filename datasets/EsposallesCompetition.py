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
def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]
import sys
import os
from keras import backend as K
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

batch_size=config.batch_size
im_height=config.im_height
im_width=config.im_width

def PIL2array(img):
    return np.array(img.getdata(),
                    np.uint8).reshape(img.size[1], img.size[0], 3)


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
        self.r_id=self.register_iterator.next()
        self.word_iterator=iter(self.reg_dict[self.r_id])
        self.w_id=-1
        self.epoch=0
        self.numberSamples=len(self.labels.keys())

    def readNormalizedImage(self,imageid):
        v=imageid.split('_');
        filename=self.DataDir+'/'+'_'.join(v[0:2])+'/words/'+imageid+".png"
        #img=1.-cv2.imread(filename,cv2.IMREAD_GRAYSCALE)/255.
        im=Image.open(filename)

        #Paste image in fixed size background
        maxsize = (config.im_height,config.im_width)
        im.thumbnail(maxsize)
        background=Image.fromarray(np.zeros(maxsize))
        background.paste(im)
        im=np.array(background)

        return im
        #return np.pad(img,(max(0,30-height),max(0,30-width)),'constant')

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
    def get_example(self):
        #minx=30
        #miny=30
        #return img
        #while True:
        if self.level=='word':
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
            self.word_iterator=iter(self.reg_dict[self.r_id])
            if K.backend()=='tensorflow':
                X=[np.asarray(self.readNormalizedImage(i))[:,:,np.newaxis] for i in self.word_iterator]
            else:
                X=[np.asarray(self.readNormalizedImage(i))[np.newaxis,:,:] for i in self.word_iterator]
            self.word_iterator=iter(self.reg_dict[self.r_id])
            Y=[np.asarray(self.labeldict[self.labels[i]]) for i in self.word_iterator]
            self.word_iterator=iter(self.reg_dict[self.r_id])
            IDS=[i for i in self.word_iterator]
            try:
                self.r_id=self.register_iterator.next()
            except StopIteration: #if no more register, shuffle and get first register again
                np.random.shuffle(self.shuffled_registers)
                self.register_iterator=iter(self.shuffled_registers)
                self.r_id=self.register_iterator.next()
                self.epoch+=1
            print X[0].shape
            return X,Y,IDS


    def get_transcription_from_categorical(self,predictions):
        if self.revdict is None: self.revdict={v:k for (k,v) in self.labeldict.iteritems()}
        return [','.join([self.revdict[timestep] for timestep in sample]).rstrip(',0') for sample in predictions]



def test_input():
    ims,labs,names=EsposallesDataset(cvset='train',level='sequence').get_example()
    widths=[]
    heights=[]
    for i in range(len(ims)):
        im=ims[i]

        heights.append(im.shape[0])
        widths.append(im.shape[1])
        #im=im*255
        im=im.astype('uint8')
        print labs[i],names[i],im.shape
        im=Image.fromarray(im[:,:,0])
        im.show()
    print np.mean(heights)

    print np.mean(widths)
    '''
    im,lab,name=EsposallesDataset(cvset='train').get_example()
    im=im*255
    im=im.astype('uint8')
    im=im[0][:,:,0]
    print im,type(im),im.shape
    im=Image.fromarray(im)
    im.show()

    '''

test_input()
