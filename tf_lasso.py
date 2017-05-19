#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function
import os, sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

#######################
import tensorflow as tf
from tflearn import rtflearn, vardict

class tflasso(rtflearn):
    def _create_network(self):
        self.vars = vardict()
        self.vars.x = tf.placeholder("float", shape=[None, self.xlen])
        self.vars.y = tf.placeholder("float", shape=[None, 1])

        #def fully_connected():
        # Create Model
        self.parameters["W1"] = tf.Variable(tf.truncated_normal([1, self.xlen], stddev=0.1), name="weight")
        self.parameters["b1"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")

        self.vars.y_predicted = tf.matmul( self.vars.x, tf.transpose(self.W1)) + self.b1
        self.saver = tf.train.Saver()
        return self.vars.y_predicted

    def _create_loss(self):
        self.train_time = tf.placeholder(tf.bool, name='train_time')
        # Minimize the squared errors
        l2_loss = tf.reduce_mean(tf.pow( self.vars.y_predicted - self.vars.y, 2))
        l2_sy = tf.scalar_summary( "L2_loss", l2_loss )
        "Lasso penalty"
        l1_penalty = tf.reduce_sum((tf.abs(tf.concat(1, [self.W1,self.b1], name = "l1" ) )) )
        l1p_sy =  tf.scalar_summary( "L1_penalty" , l1_penalty )
        "total"
        tot_loss = l2_loss + self.ALPHA * l1_penalty
        tot_loss_sy =  tf.scalar_summary( "loss" , tot_loss )
        "R2"
        _, y_var = tf.nn.moments(self.vars.y, [0,1])
        rsq =  1 - l2_loss / y_var
        rsq_sy = tf.scalar_summary( "R2", rsq)
        return tot_loss

if __name__ == "__main__":
    datafile = "../../data/atac_tss_800_1.h5"
#    datafile = "../data/test.h5"
    paramfile = "tflasso_tss_100"
    with pd.HDFStore(datafile) as store:
        print(store.groups())
        y_ = store["y"]
        X_ = store["X"]

    """ transform data """
    sys.path.append("../")
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

    tfl = tflasso(ALPHA = 2e-1, dropout = False )
    tfl.fit( train_X, train_Y )



