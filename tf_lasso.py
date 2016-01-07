#!/usr/bin/env python3
# coding: utf-8
import os, sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt
import joblib , dill

#######################
def batchgen(batchsize):
    
    def getbatch(x,y):
        assert (len(x) == len(y)), "dimension mismatch"
        for i in range(0, len(y), batchsize):
            yield x[i:i+batchsize], y[i:i+batchsize], 
    return getbatch

#######################
class vardict(dict):
    #__module__ = os.path.splitext(os.path.basename(__file__))[0]  ### look here ###
    def __init__(self, *args, **kwargs):
                dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        return self[name]

    def __setattr__(self,name, val):
        self.__dict__[name] = val

    def __getstate__(self):
        return self.__dict__.items()

    def __setstate__(self, items):
        for key, val in items:
            self.__dict__[key] = val


import tensorflow as tf

class tflasso():
   def _create_network(self):
        self.vars = vardict()
        self.vars.xx = tf.placeholder("float", shape=[None, self.xlen])
        self.vars.yy = tf.placeholder("float", shape=[None, 1])

        # Create Model
        self.parameters["W1"] = tf.Variable(tf.truncated_normal([1, self.xlen], stddev=0.1), name="weight")
        self.parameters["b1"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
        self.vars.y_predicted = tf.matmul( self.vars.xx, tf.transpose(self.W1)) + self.b1
        self.saver = tf.train.Saver()
        
    def _create_loss(self):
        # Minimize the squared errors
        l2_loss = tf.reduce_mean(tf.pow( self.vars.y_predicted - self.vars.yy, 2))
        # Lasso penalty
        l1_penalty = tf.reduce_sum((tf.abs(tf.concat(1, [self.W1,self.b1], name = "l1" ) )) )
        tot_loss = l2_loss + self.ALPHA * l1_penalty
        return tot_loss
        
       
if __name__ == "__main__":
    datafile = "../data/atac_tss_800_1.h5"
#    datafile = "../data/test.h5"
    paramfile = "tflasso_tss_100"
    with pd.HDFStore(datafile) as store:
        print(store.groups())
        y_ = store["y"]
        X_ = store["X"]

    """ transform data """
    from transform_tss import safelog, sumstrands, groupcolumns

    feature_step = 100
    select = list(feature_step * np.arange(-2,3,1))

    Xgr = groupcolumns(X_, step = feature_step,select = select)

    X, y = safelog(Xgr, y_)

    from sklearn.preprocessing import PolynomialFeatures
    pf3 = PolynomialFeatures(degree=3)
    X3 = pf3.fit_transform(X)
    trainsamples = 4000
    train_X, train_Y = X3[:trainsamples], y[:trainsamples].as_matrix()

    import json

    tfl = tflasso(ALPHA = 2e-1)
    tfl.fit( train_X, train_Y )



