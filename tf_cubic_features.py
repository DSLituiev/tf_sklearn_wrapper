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
    def __init__(self,
             learning_rate = 2e-2,
             training_epochs = 5000,
                display_step = 100,
                BATCH_SIZE = 100,
                ALPHA = 1e-4,
                NUM_CORES = 3,
                checkpoint_dir = "./checkpoints/"
         ):
        self.learning_rate = learning_rate
        self.training_epochs=training_epochs
        self.display_step = display_step
        self.ALPHA = ALPHA
        self.BATCH_SIZE = BATCH_SIZE
        self.checkpoint_dir = checkpoint_dir
        self.NUM_CORES = NUM_CORES  # Choose how many cores to use.
        
        self.parameters = vardict()
#     def __getattr__(self, name):
#         return self.parameters[name]
    def __getattr__(self, key):
        if key.startswith('__') and key.endswith('__'):
            return super(tflasso, self).__getattr__(key)
        return self.__getitem__(key)

    def __getitem__(self, key):
        #if hasattr(self, "parameters") and
        if key in self.parameters:
            return self.parameters[key]
        else:
            print(key, "not found", file = sys.stderr)
            return 

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
        #l1_penalty =(tf.reduce_sum(tf.abs(self.W1)) +  tf.reduce_sum(tf.abs(self.b1)))/ (1+sum([int(x) for x in self.W1.get_shape()]))
        l1_penalty = tf.reduce_sum((tf.abs(tf.concat(1, [self.W1,self.b1], name = "l1" ) )) )
        tot_loss = l2_loss + self.ALPHA * l1_penalty
        return tot_loss
        
    def get_params(self, load = True):
        params = {}
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            with tf.Session(config = sess_config) as sess:
                if load:
                    self._load_(sess)
                for kk, vv in self.parameters.items():
                    params[kk] = vv.eval()
        return params
        
    def _load_(self, sess, checkpoint_dir = None):
        if checkpoint_dir:
            self.checkpoint_dir = checkpoint_dir
        
        print("loading a session")
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise Exception("no checkpoint found")
        #print( "loaded b1:",  self.parameters.b1.name , self.parameters.b1.eval()[0][0]  , sep = "\t" )
        assert self.xlen == int(self.vars.xx.get_shape()[1]), "dimension mismatch"
        return
    
    def transform(self, X, y = None, load = True):
        self.xlen = X.shape[1]
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            with tf.Session(config = sess_config) as sess:
                if load:
                    self._load_(sess)

                y_predicted = sess.run( self.vars.y_predicted,
                                feed_dict = { self.vars.xx: X})
                if y is not None:
                    tot_loss = self._create_loss()
                    self.loss = sess.run( tot_loss,
                                    feed_dict = { self.vars.xx: X, self.vars.yy :  np.reshape(y, [-1, 1]) })
        return y_predicted
    
    def fit(self, train_X, train_Y , load = True):
        #self.X = train_X
        self.xlen = train_X.shape[1]
        self.r2_progress = []
        # n_samples = y.shape[0]
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            tot_loss = self._create_loss()
            optimizer = tf.train.AdagradOptimizer( self.learning_rate).minimize(tot_loss)

            # Initializing the variables
            init = tf.initialize_all_variables()
            " training per se"
            getb = batchgen( self.BATCH_SIZE)

            yvar = train_Y.var()
            print(yvar)
            # Launch the graph
        
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            with tf.Session(config= sess_config) as sess:
                sess.run(init)
                if load:
                    try:
                        self._load_(sess)
                    except Exception as ex:
                        print( "loading failed", file = sys.stderr)
                        print(ex, file = sys.stderr)
                # Fit all training data
                for epoch in range( self.training_epochs):
                    for (_x_, _y_) in getb(train_X, train_Y):
                        _y_ = np.reshape(_y_, [-1, 1])
                        sess.run(optimizer, feed_dict={ self.vars.xx: _x_, self.vars.yy: _y_})
                    # Display logs per epoch step
                    if (1+epoch) % self.display_step == 0:
                        cost = sess.run(tot_loss,
                                feed_dict={ self.vars.xx: train_X,
                                        self.vars.yy: np.reshape(train_Y, [-1, 1])})
                        #rsq = sess.run(rsquared, feed_dict={xx: train_X, yy: np.reshape(train_Y, [-1, 1])})
                        rsq =  1 - cost / yvar
                        self.r2_progress.append( (epoch, rsq))
                        logstr = "Epoch: {:4d}\tcost = {:.4f}\tR^2 = {:.4f}".format((epoch+1), cost, rsq)
                        print(logstr, file = sys.stderr )
                        self.saver.save(sess, self.checkpoint_dir + 'model.ckpt',
                           global_step= 1+ epoch)
                        #print("\tb1",  self.parameters.b1.name , self.parameters.b1.eval()[0][0] , sep = "\t")
                        #print( "W=", sess.run(W1))  # "b=", sess.run(b1)
                print("Optimization Finished!", file = sys.stderr)
#                 print("cost = ", sess.run( tot_loss , feed_dict={self.vars.xx: train_X, self.vars.yy: np.reshape(train_Y, [-1, 1]) }) )
#                 print("W1 = ", sess.run(self.parameters.W1), )
#                 print("b1 = ", sess.run(self.parameters.b1) )
        return self

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



