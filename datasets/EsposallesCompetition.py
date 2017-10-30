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


batch_size=1
im_height=80
im_width=125

class EsposallesDataset():
    def __init__(self,BaseDir='/home/ntoledo/datasets/OfficialEsposalles',cvset='train',level='word'):
        self.BaseDir=BaseDir
        self.DataDir=BaseDir+'/'+cvset
        self.GroundTruth=self.DataDir+'/groundtruth_full.txt'
        self.level=level
        self.transcriptions={}
        self.categories={}
        self.persons={}
        with open(self.GroundTruth, mode='r') as infile:
            for line in infile:
                values=line.rstrip('\n').split(':',4)
                self.transcriptions[values[0]]=values[1]
                self.categories[values[0]]=values[2]
                self.persons[values[0]]=values[3]
        self.categorydict={l:n for (n,l) in enumerate(['other','surname','name','location','occupation','state'])}
        self.revcategorydict={v:k for (k,v) in self.categorydict.iteritems()}
        self.persondict={l:n for (n,l) in enumerate(['none','other_person','husband','wife','husbands_father','husbands_mother','wifes_mother','wifes_father'])}
        self.revpersondict={v:k for (k,v) in self.persondict.iteritems()}
        self.generate_previous_categories_and_regdict()
        self.shuffled_registers=[k for k in self.reg_dict.iterkeys()]
        self.shuffled_registers.sort()
        self.register_iterator=iter(self.shuffled_registers)
        self.epoch_size = len(self.shuffled_registers)
        self.r_id=self.register_iterator.next()
        self.epoch=0

    def readNormalizedImage(self,imageid):
        # Read image from file
        v=imageid.split('_');
        filename=self.DataDir+'/'+'_'.join(v[0:2])+'/words/'+imageid+".png"
        im=Image.open(filename)

        #Resize image to fit maximum size
        maxsize = (im_height,im_width)
        im.thumbnail(maxsize)

        #Create background to paste image into
        background=np.zeros(maxsize)
        #average_bckg_color=int(np.mean(im[np.where(im<100)]))
        average_bckg_color = int(np.mean(np.array(im)))
        background.fill(average_bckg_color)
        background=Image.fromarray(background)

        #Paste image centered into grey background
        background.paste(im,box=(maxsize[1]/2-im.size[0]/2,maxsize[0]/2-im.size[1]/2))
        im=np.array(background)
        im= 1.-np.array(im)/255.

        return im


    def generate_previous_categories_and_regdict(self):
        l=self.categories.keys()
        l=sorted(l,key=natural_key)
        prevcategory=6
        prevpag,prevreg,pag,reg=(0,0,0,0)
        self.prevcategories={}
        self.reg_dict={}
        for s in l:
            m=re.search('^idPage([0-9]{5})_Record([0-9]{1,2})_Line([0-9]{1,2})_Word([0-9]{1,2})$',s);
            pag,reg,lin,pos=int(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4))
            if self.reg_dict.has_key((pag,reg)):
               self.reg_dict[pag,reg].append(s)
            else:
               self.reg_dict[pag,reg]=[s]
            if prevpag==0 or prevpag==pag and prevreg == reg:
               self.prevcategories[s]=prevcategory
               prevcategories=self.categorydict[self.categories[s]]
            else:
               prevcategory=6
               self.prevcategories[s]=prevcategory
            prevpag=pag
            prevreg=reg

    def get_batch(self,show_word_ids=False):

        IDS=[i for i in self.reg_dict[self.r_id]]
        X=[np.asarray(self.readNormalizedImage(i))[:,:,np.newaxis] for i in IDS ]
        category=[np.asarray(self.categorydict[self.categories[i]]) for i in IDS ]
        person=[np.asarray(self.persondict[self.persons[i]]) for i in IDS ]

        try:
            self.r_id=self.register_iterator.next()
        except StopIteration: #if no more register, shuffle and get first register again
            np.random.shuffle(self.shuffled_registers)
            self.register_iterator=iter(self.shuffled_registers)
            self.r_id=self.register_iterator.next()
            self.epoch+=1

        X=np.stack(X)
        X=X[np.newaxis,:,:,:,:]
        category=np_utils.to_categorical(category, len(self.categorydict.keys()))
        category=category[np.newaxis,:,:]
        person=np_utils.to_categorical(person, len(self.persondict.keys()))
        person=person[np.newaxis,:,:]

        return X,category,person,[IDS]


    def get_labels_from_categorical(self,ids,categories,persons=None):
        if persons is None:
            return [ zip(r_ids,[ self.revcategorydict[np.argmax(sample)] for sample in r_categories]) for r_ids,r_categories in zip (ids,categories)]
        return [ zip(r_ids,[ self.revcategorydict[np.argmax(sample)] for sample in r_categories],[ self.revpersondict[np.argmax(sample)] for sample in r_persons]) for r_ids,r_categories,r_persons in zip (ids,categories,persons)]

    def show_batch(self):

        ims,cats,pers,names=self.get_batch()
        widths=[]
        heights=[]

        for i in range(len(ims[0,:,:,:,0])):
            im=ims[0,i,:,:,0]

            print im.shape

            im=im.astype('uint8')
            im=Image.fromarray(im)
            im.show()
            plt.imshow(im)


            print names[i],cats[0,i,:],pers[0,i,:]
            plt.show()

def main():
    E=EsposallesDataset()
    E.show_batch()

if __name__=="__main__":
    main()
