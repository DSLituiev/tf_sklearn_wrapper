#!/usr/bin/env python3
# coding: utf-8
import os, sys
from tqdm import tqdm
import numpy  as np
#import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf

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
#######################
def summary_dict(summary_str, summary_proto = None):
    "convert summary string to a dictionary"
    if summary_proto is None:
        summary_proto = tf.Summary()
    summary_proto.ParseFromString(summary_str)
    summaries = {}
    for val in summary_proto.value:
        # Assuming all summaries are scalars.
        summaries[val.tag] = val.simple_value
    return summaries
#######################
class tflearn():
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
        raise NotImplementedError              
        
    def _create_loss(self):
        with tf.name_scope("loss") as scope:
            # Minimize the squared errors
            l2_loss = tf.reduce_mean(tf.pow( self.vars.y_predicted - self.vars.yy, 2))
            l2_sy = tf.scalar_summary( "L2_loss", l2_loss )
            # Lasso penalty
            #l1_penalty = tf.reduce_sum((tf.abs(tf.concat(1, [self.W1,self.b1]) )) )
            #l1p_sy =  tf.scalar_summary( "L1_penalty" , l1_penalty )
            tot_loss = l2_loss #+ self.ALPHA * l1_penalty
            tot_loss_sy =  tf.scalar_summary( "loss" , tot_loss )
            
            _, y_var = tf.nn.moments(self.vars.yy, [0,1])
            rsq =  1 - l2_loss / y_var
            rsq_sy = tf.scalar_summary( "R2", rsq)
            
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
        
        print("loading a session", file = sys.stderr)
        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        # print("checkpoint:", ckpt, file = sys.stderr)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print(ckpt, file = sys.stderr)
            raise Exception("no checkpoint found")
        #print( "loaded b1:",  self.parameters.b1.name , self.parameters.b1.eval()[0][0]  , sep = "\t" )
        assert self.xlen == int(self.vars.xx.get_shape()[1]), "dimension mismatch"
        self.last_ckpt_num = int(ckpt.all_model_checkpoint_paths[-1].split("-")[-1])
        return ckpt
    
    def transform(self, X, y = None, load = True):
        self.train = False
        self.xlen = X.shape[1]
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            # Initializing the variables
            init = tf.initialize_all_variables()

            with tf.Session(config = sess_config) as sess:
                if load:
                    self._load_(sess)
                else:
                    sess.run(init)

                y_predicted = sess.run( self.vars.y_predicted,
                                feed_dict = { self.vars.xx: X})
                if y is not None:
                    tot_loss = self._create_loss()
                    self.loss = sess.run( tot_loss,
                                    feed_dict = { self.vars.xx: X, self.vars.yy :  np.reshape(y, [-1, 1]) })
        return y_predicted

    def fit(self, train_X, train_Y , load = True):
        self.last_ckpt_num = 0
        self.train = True
        #self.X = train_X
        self.xlen = train_X.shape[1]
        self.r2_progress = []
        yvar = train_Y.var()
        print(yvar)
        # n_samples = y.shape[0]
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            
            tot_loss = self._create_loss()
            train_op = tf.train.AdagradOptimizer( self.learning_rate).minimize(tot_loss)
            # Merge all the summaries and write them out
            summary_op = tf.merge_all_summaries()

            # Initializing the variables
            init = tf.initialize_all_variables()
            " training per se"
            getb = batchgen( self.BATCH_SIZE)

            
            # Launch the graph        
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            with tf.Session(config= sess_config) as sess:
                sess.run(init)
                if load:
                    self._load_(sess)
                # write summaries out
                summary_writer = tf.train.SummaryWriter("./tmp/mnist_logs", sess.graph_def)
                summary_proto = tf.Summary()
                # Fit all training data
                for epoch in tqdm(range( self.last_ckpt_num, self.last_ckpt_num + self.training_epochs)):
                    "do minibatches"
                    for (_x_, _y_) in getb(train_X, train_Y):
                        _y_ = np.reshape(_y_, [-1, 1])                        
                        feed_dict={ self.vars.xx: _x_, self.vars.yy: _y_, self.vars.keep_prob : 0.5}
                        sess.run(train_op, feed_dict = feed_dict)
                    # Display logs once in `display_step` epochs
                    
                    if (epoch) % self.display_step == 0:
                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, epoch)
                        summary_d = summary_dict(summary_str, summary_proto)
                        #print(type(summary_str))
                        
#                         cost = sess.run(tot_loss,
#                                 feed_dict={ self.vars.xx: train_X,
#                                         self.vars.yy: np.reshape(train_Y, [-1, 1])})
#                         #rsq = sess.run(rsquared, feed_dict={xx: train_X, yy: np.reshape(train_Y, [-1, 1])})
#                         rsq =  1 - cost / yvar
                        self.r2_progress.append( (epoch, summary_d["R2"]))
                        logstr = "Epoch: {:4d}\ttot loss= {:.4f}\tL2 loss= {:.4f}\tR^2= {:.4f}".format((epoch+1), 
                                        summary_d["loss"], summary_d["L2_loss"], summary_d["R2"],)
                        print(logstr, file = sys.stderr )
                        self.saver.save(sess, self.checkpoint_dir + '/' +'model.ckpt',
                           global_step=  epoch)
                        self.last_ckpt_num = epoch
                        #0print("\tb1",  self.parameters.b1.name , self.parameters.b1.eval()[0][0] , sep = "\t")
                        #print( "W=", sess.run(W1))  # "b=", sess.run(b1)
                print("Optimization Finished!", file = sys.stderr)
#                 print("cost = ", sess.run( tot_loss , feed_dict={self.vars.xx: train_X, self.vars.yy: np.reshape(train_Y, [-1, 1]) }) )
#                 print("W1 = ", sess.run(self.parameters.W1), )
#                 print("b1 = ", sess.run(self.parameters.b1) )
        return self

