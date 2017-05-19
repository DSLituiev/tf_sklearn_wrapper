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


flags = tf.app.flags
FLAGS = flags.FLAGS

# define flags (note that Fomoro will not pass any flags by default)
flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

# define artifact directories where results from the session can be saved
model_path = os.environ.get('MODEL_PATH', 'models/')
checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

class tf_3step_ar(rtflearn):
    def _create_network(self):
        self.vars = vardict()
        self.vars.x = tf.placeholder("float", shape=[None, self.xlen], name = "x")
        self.vars.y = tf.placeholder("float", shape=[None, 3], name = "y")

        #def fully_connected():
        # Create Model
        self.parameters["decay_rate"] = tf.Variable(tf.constant(0.1, shape=[1]), 
                                                    name="decay_rate")
        self.parameters["synthesis_rate"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), 
                                                        name="synthesis_rate")

        self.parameters["z0"] = tf.Variable(tf.truncated_normal(
                                [self.n_samples, 1], stddev=0.1), name="z0")

        self.parameters["effect_size_act"] = tf.Variable(tf.truncated_normal(
                                [ self.xlen], stddev=0.1), name="effect_size")
        #self.parameters["ar_design_matrix"] = tf.concat( [np.zeros((self.N, self.N)), np.eye(self.N), np.eye(self.N)],)
        #tf.Variable(tf.truncated_normal([1, self.xlen], stddev=0.1), name="effect_size")

        # A: [ 1 x 3 ]
        A = tf.concat(0, [tf.Variable(tf.constant(1.0, shape=[1])),
                          tf.exp( - self.parameters["decay_rate"]),
                          tf.exp( - 2 * self.parameters["decay_rate"])
                         ], name="A" )
        # decay: [ N x 3 ]
        carryover_amount = A * self.parameters["z0"]
        # synthesis: [ N x 3 ]
        new_amount = (1-A) * tf.matmul(self.vars.x,
                                       tf.transpose(self.parameters["effect_size_act"]) )

        self.vars.y_predicted = carryover_amount + new_amount
        self.saver = tf.train.Saver()
        return self.vars.y_predicted

    def _create_loss(self):
        # Minimize the squared errors
        l2_loss = tf.reduce_mean(tf.pow( self.vars.y_predicted - self.vars.y, 2))
        l2_sy = tf.scalar_summary( "L2_loss", l2_loss )
        "Lasso penalty"
        #l1_penalty = tf.reduce_sum( tf.abs( self.parameters["effect_size_act"] ) )
        l1_penalty = tf.reduce_sum( tf.abs(
            self.parameters["z0"] - tf.reduce_mean(self.parameters["z0"]) ) )
        tf.scalar_summary( "L1_penalty" , l1_penalty )
        "total"
        tot_loss = l2_loss + self.ALPHA * l1_penalty
        tf.scalar_summary( "loss" , tot_loss )
        "R2"
        _, y_var = tf.nn.moments(self.vars.y, [0,1])
        rsq =  1 - l2_loss / y_var
        tf.scalar_summary( "R2", rsq)
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

    tfl = tflasso(ALPHA = 2e-6, dropout = False )
    tfl.fit( train_X, train_Y )



