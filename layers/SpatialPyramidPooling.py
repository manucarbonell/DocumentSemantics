# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 15:49:14 2016

@author: jtoledo

"""

from keras.engine import Layer
import theano.tensor as T
from theano.sandbox.cuda.dnn import GpuDnnPool

class SPP (Layer):
    def __init__(self, bins):
        self.bins = bins
        super(SPP,self).__init__()
    def build(self,input_shape):
        self.ax=input_shape[-2]
        self.ay=input_shape[-1]
        super(SPP,self).__init__()
    def call(self, x, mask=None):
        output=[]
        for (binsize_x,binsize_y) in self.bins:
            print "OPERAND TYPES",type(x.shape[-2:]),type(map(float,(binsize_x,binsize_y)))
            wxy=(x.shape[-2:]/map(float,(binsize_x,binsize_y))).ceil().astype('int32')
            remainder=wxy*[binsize_x,binsize_y] - x.shape[-2:]
            pxy=((remainder+1)/2.0).astype('int32')

            max_pool=GpuDnnPool(mode='max')
            output.append(max_pool(img=x,
                                   ws=wxy,
                                   stride=wxy,
                                   pad=pxy).flatten(ndim=2))
        return T.concatenate(output, axis=1)
    def compute_output_shape(self,input_shape):
        return tuple((input_shape[0],sum([ a*b for (a,b) in self.bins])*input_shape[-3]))
