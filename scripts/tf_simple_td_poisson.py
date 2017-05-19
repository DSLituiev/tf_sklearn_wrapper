#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function
import os, sys
import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

#######################
import tensorflow as tf
from tflearn import rtflearn, vardict, batch_norm

class tf_td_poisson(rtflearn):
    def _create_network(self):
        self.vars = vardict()
        self.train_time = tf.placeholder(tf.bool, name='train_time')
        self.vars.x = tf.placeholder("float", shape=[None, self.xlen], name = "x")
        self.vars.y = tf.placeholder("float", shape=[None, 1], name = "y")

        self.vars.x = batch_norm(self.vars.x, self.train_time)
        tfs, gts = tf.unpack(tf.transpose(self.vars.x), num=2)
        print("tfs", tfs.get_shape())

        # Create Model
        self.parameters["neg_penalty_const"] = tf.Variable(tf.constant(1e1, shape=[]),
                                                     name="neg_penalty_const", trainable = False)
        self.parameters["synthesis_interact"] = tf.Variable(tf.constant(10.0, shape=[1,]),
                                                        name="synthesis_interact")

        self.parameters["synthesis_tf_only"] = tf.Variable(tf.constant(1.0, shape=[1,]),
                                                        name="synthesis_tf_only")

        self.parameters["synthesis_const"] = tf.Variable(tf.constant(4.0, shape=[1,]),
                                                        name="synthesis_const")

        #self.parameters["ar_design_matrix"] = tf.concat( [np.zeros((self.N, self.N)), np.eye(self.N), np.eye(self.N)],)
        #tf.Variable(tf.truncated_normal([1, self.xlen], stddev=0.1), name="effect_size")

        # lambda:
        self.vars.y_predicted = tf.transpose(
                            self.parameters["synthesis_const"] + \
                                tfs * (
                                    self.parameters["synthesis_tf_only"] + self.parameters["synthesis_interact"] * gts \
                                      )
                                            )
        self.vars.y_predicted = tf.reshape(self.vars.y_predicted, [-1, 1])

        #self.vars.y_predicted = gts * 1e-2
        self.saver = tf.train.Saver()
        return self.vars.y_predicted

    def _ydiff(self):
        print( "y_predicted", self.vars.y_predicted.get_shape() )
        print( "y", self.vars.y.get_shape())
        return self.vars.y_predicted - self.vars.y

    def _create_loss(self):
        # Minimize the squared errors
        print("loss")
        epsilon = 1e-5
        y_pred_pos = tf.nn.relu( self.vars.y_predicted)
        y_pred_neg_penalty = tf.reduce_mean( tf.nn.relu( - self.vars.y_predicted), name = "neg_penalty")

        poisson_loss = tf.reduce_mean(tf.abs(y_pred_pos) - self.vars.y * tf.log(1 + tf.abs(y_pred_pos)),
                                      name = "poisson_loss")
        tf.scalar_summary("poisson_loss", poisson_loss )
        tf.scalar_summary( "neg_penalty", y_pred_neg_penalty )

        tot_loss = poisson_loss + self.parameters["neg_penalty_const"] * y_pred_neg_penalty

        l2_loss = tf.reduce_mean(tf.pow( self.vars.y_predicted - self.vars.y, 2))
        tf.scalar_summary( "loss" , tot_loss )
        #tf.scalar_summary( "y[0]" , self.vars.y_predicted[9] )
        #tf.scalar_summary( "y_hat[0]" , self.vars.y[9,0] )
        tf.scalar_summary( "l2_loss" , l2_loss )
        "R2"
        _, y_var = tf.nn.moments(self.vars.y, [0,1])
        rsq =  1 - l2_loss / y_var
        tf.scalar_summary( "R2", rsq)
        return  tot_loss

if __name__ == "__main__":

    flags = tf.app.flags
    #FLAGS = flags.FLAGS

    # define flags (note that Fomoro will not pass any flags by default)
    flags.DEFINE_boolean('skip-training', False, 'If true, skip training the model.')
    flags.DEFINE_boolean('restore', False, 'If true, restore the model from the latest checkpoint.')

    # define artifact directories where results from the session can be saved
    model_path = os.environ.get('MODEL_PATH', 'models/')
    checkpoint_path = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
    summary_path = os.environ.get('SUMMARY_PATH', 'logs/')

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



