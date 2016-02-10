#!/usr/bin/env python3
# coding: utf-8
from __future__ import print_function
import os, sys
from tqdm import tqdm
import numpy  as np
#import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
import logging

#######################
def batchgen(batchsize, dictionary = False):
    
    def getbatch( *args):
        """ generate batch train tuple from all arguments """
        if type(args[-1]) is list:
            ylen = len(y)
        else:
            ylen = args[-1].shape[0]

        if len(args) > 1:
            for a in args:
                assert (a.shape[0] == ylen ), "dimension mismatch"

        for i in range(0, ylen, batchsize):
            yield (a[i:i+batchsize] for a in args)

    def getbatchdict( **kwargs):
        """ generate batch feed dictionary from a key-value pairs """
        if type(kwargs.values()[-1]) is list:
            ylen = len(y)
        else:
            ylen = kwargs.values()[-1].shape[0]

        if len(args) > 1:
            for kk, vv in kwargs.items():
                assert (vv.shape[0] == ylen ), "dimension mismatch"

        for i in range(0, ylen, batchsize):
            yield {kk : vv[i:i+batchsize] for kk, vv in kwargs}

    if dictionary:
        return getbatchdict
    else:
        return getbatch

#######################
class vardict(dict):
    #__module__ = os.path.splitext(os.path.basename(__file__))[0]  ### look here ###
    def __init__(self, *args, **kwargs):
                dict.__init__(self, *args, **kwargs)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("key %s not found" % name )

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
    def __init__(self, *args,
        **kwargs      ):
        defaults = dict(
                learning_rate = 2e-2,
                epochs = 5000,
                display_step = 100,
                BATCH_SIZE = 100,
                ALPHA = 1e-4,
                NUM_CORES = 3,
                checkpoint_dir = "./checkpoints/",
                dropout = 0.5,
                optimizer = tf.train.AdagradOptimizer
                )
        for kk,vv in defaults.items():
            if kk not in kwargs:
                kwargs[kk] = vv
        for kk,vv in kwargs.items():
            setattr(self, kk, vv)
        self.parameters = vardict()

        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
            print("key", key, "not found", file = sys.stderr)
            return 

    def _create_network(self):
        print( """
        # Example:

        # create placeholders 
        self.vars = vardict()
        self.vars.x = tf.placeholder("float", shape=[None, self.xlen])
        self.vars.y = tf.placeholder("float", shape=[None, 1])

        # create parameter variables
        self.parameters["W1"] = tf.Variable(tf.truncated_normal([1, self.xlen], stddev=0.1), name="weight")
        self.parameters["b1"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")

        # Create Model
        self.vars.y_predicted = tf.matmul( self.vars.x, tf.transpose(self.W1)) + self.b1
        self.saver = tf.train.Saver()

        return self.vars.y_predicted
        """ ,file = sys.stderr)
        raise NotImplementedError              
         
           

    def _get_summary_keys_(self):
        #summaries = {}
        sumtags = []
        sess = tf.Session()

        all_summary_tensors = tf.get_collection(tf.GraphKeys.SUMMARIES)

        for summary_t in all_summary_tensors:
            tag_input = summary_t.op.inputs[0]  # The tag input is the 0th input.
            tags = sess.run(tag_input)

            if isinstance(tags, str):
                sumtags.append( tags )
            else:
                for tag in tags.flatten():
                    sumtags.append(tag) 
        return sumtags 
        
    def _create_loss(self):
        """
        define loss variable and summaries;
        the method must return a tf.Variable (not a summary!)
        """
        raise NotImplementedError              
       
    def get_params(self, load = True):
        params = {}
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            sess_config = tf.ConfigProto(inter_op_parallelism_threads= 1,
                                       intra_op_parallelism_threads= 1)
            with tf.Session(config = sess_config) as sess:
                if load:
                    self._load_(sess)
                else:
                    # Initializing the variables
                    init = tf.initialize_all_variables()
                    sess.run(init)
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
            raise IOError("no checkpoint found")
        #print( "loaded b1:",  self.parameters.b1.name , self.parameters.b1.eval()[0][0]  , sep = "\t" )
        assert self.xlen == int(self.vars.x.get_shape()[1]), "dimension mismatch"
        self.last_ckpt_num = int(ckpt.all_model_checkpoint_paths[-1].split("-")[-1])
        return ckpt

    def predict(self, X, y = None, load = True, debug = False):
        self.train = False
        if len(X.shape) > 1:
            self.xlen = X.shape[1]
        else:
            self.xlen = 1
        g = tf.Graph()
        with g.as_default():
            "fetch a placeholder of the predicted variable"
            ph_y_predicted = self._create_network()
            if not ("keep_prob" in self.vars or hasattr( self.vars, "keep_prob") ):
                self.dropout = 0.0
            tot_loss = self._create_loss()
            summary_op = tf.merge_all_summaries()
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            # Initializing the variables
            init = tf.initialize_all_variables()

            with tf.Session(config = sess_config) as sess:
                if load:
                    self._load_(sess)
                else:
                    sess.run(init)

                feed_dict={ self.vars.x: X, }
                if self.dropout > 0 and self.dropout < 1:
                    feed_dict[ self.vars.keep_prob] = self.dropout 
                
                y_predicted = sess.run( self.vars.y_predicted,
                                feed_dict = feed_dict )
                if debug:
                    for kk, vv in self.vars.items():
                        if vv not in feed_dict and kk != "y" and kk!="yy":
                            try:
                                print(kk, sess.run( vv, feed_dict = feed_dict ).shape )
                            except:
                                print( "unable to evaluate %s" % kk )
                                pass

                if y is not None:
                    feed_dict[ self.vars.y ] = np.reshape(y, [-1, 1])
                    self.summary_proto = tf.Summary()
                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_d = summary_dict(summary_str, self.summary_proto)

                    summary_plainstr =  "\t".join(["{:s}: {:.4f}".format(k,v) for k,v in summary_d.items() ])
                    print( summary_plainstr, file = sys.stderr )

                    self.loss = sess.run( tot_loss,
                                    feed_dict = { self.vars.x: X, self.vars.y :  np.reshape(y, [-1, 1]) })
        return y_predicted

    def fit(self, train_X, train_Y , test_X= None, test_Y = None, load = True, epochs = None):
        if epochs:
            self.epochs = epochs
        self.last_ckpt_num = 0
        self.train = True
        #self.X = train_X
        self.xlen = train_X.shape[1]
        self.r2_progress = []
        self.train_summary = []
        self.test_summary = []
        yvar = train_Y.var()
        #print("variance(y) = ", yvar, file = sys.stderr)
        # n_samples = y.shape[0]
        g = tf.Graph()
        with g.as_default():
            self._create_network()
            if not ("keep_prob" in self.vars or hasattr( self.vars, "keep_prob") ):
                self.dropout = 0.0
            tot_loss = self._create_loss()
            train_op = self.optimizer( self.learning_rate).minimize(tot_loss)
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
                    try:
                        self._load_(sess)
                    except IOError as ex:
                        print(ex, file = sys.stderr)
                else:
                    if not os.path.exists(self.checkpoint_dir):
                        os.makedirs(self.checkpoint_dir)
                # write summaries out
                summary_writer = tf.train.SummaryWriter("./tmp/mnist_logs", sess.graph_def)
                summary_proto = tf.Summary()
                # Fit all training data
                print("training epochs: %u ... %u, saving each %u' epoch" % \
                        (self.last_ckpt_num, self.last_ckpt_num + self.epochs, self.display_step),
                        file = sys.stderr)
                for macro_epoch in tqdm(range( self.last_ckpt_num//self.display_step ,
                                         (self.last_ckpt_num + self.epochs)//  self.display_step )):
                    "do minibatches"
                    for subepoch in tqdm(range(self.display_step)):
                        for (_x_, _y_) in getb(train_X, train_Y):
                            _y_ = np.reshape(_y_, [-1, 1])                        
                            if self.dropout:
                                feed_dict={ self.vars.x: _x_, self.vars.y: _y_, self.vars.keep_prob : self.dropout}
                            else:
                                feed_dict={ self.vars.x: _x_, self.vars.y: _y_ }
                            sess.run(train_op, feed_dict = feed_dict)
                    epoch = macro_epoch * self.display_step
                    # Display logs once in `display_step` epochs
                    
                    _sets_ = ["train"]
                    _xs_ = [ train_X ]
                    _ys_ = [ train_Y ]
                    summaries = {}
                    summaries_plainstr = []
                    if (test_X is not None) and (test_Y is not None):
                        _sets_.append("test")
                        _xs_.append( test_X )
                        _ys_.append( test_Y )

                    for _set_, _x_, _y_ in zip(_sets_, _xs_, _ys_ ):
                        _y_ = np.reshape(_y_, [-1, 1])                        

                        feed_dict={ self.vars.x: _x_, self.vars.y: _y_ }
                        if self.dropout:
                            feed_dict[ self.vars.keep_prob ] = self.dropout 

                        summary_str = sess.run(summary_op, feed_dict=feed_dict)
                        summary_writer.add_summary(summary_str, epoch)
                        summary_d = summary_dict(summary_str, summary_proto)
                        summaries[_set_] = summary_d

                        #summary_d["epoch"] = epoch

                        self.r2_progress.append( (epoch, summary_d["R2"]))
                        summaries_plainstr.append(  "\t".join(["",_set_] +["{:s}: {:.4f}".format(k,v) if type(v) is float else "{:s}: {:s}".format(k,v) for k,v in summary_d.items() ]) )

                    self.train_summary.append( summaries["train"] )
                    if  "test" in summaries:
                        self.test_summary.append( summaries["test"] )

                    logstr = "Epoch: {:4d}\t".format(epoch) + "\n"+ "\n".join(summaries_plainstr)
                    print(logstr, file = sys.stderr )
                    self.saver.save(sess, self.checkpoint_dir + '/' +'model.ckpt',
                       global_step=  epoch)
                    self.last_ckpt_num = epoch
                        #0print("\tb1",  self.parameters.b1.name , self.parameters.b1.eval()[0][0] , sep = "\t")
                        #print( "W=", sess.run(W1))  # "b=", sess.run(b1)
                print("Optimization Finished!", file = sys.stderr)
#                 print("cost = ", sess.run( tot_loss , feed_dict={self.vars.x: train_X, self.vars.y: np.reshape(train_Y, [-1, 1]) }) )
#                 print("W1 = ", sess.run(self.parameters.W1), )
#                 print("b1 = ", sess.run(self.parameters.b1) )
        return self

###############################################
class rtflearn(tflearn):
    """
    **regression** template class for sklearn-style tensorflow wrapper
    """
    def _create_loss(self):
        """
        define loss variable and summaries;
        the method must return a tf.Variable (not a summary!)
        """
        with tf.name_scope("loss") as scope:
            # Minimize the squared errors
            l2_loss = tf.reduce_mean(tf.pow( self.vars.y_predicted - self.vars.y, 2))
            l2_sy = tf.scalar_summary( "L2_loss", l2_loss )
            # Lasso penalty
            #l1_penalty = tf.reduce_sum((tf.abs(tf.concat(1, [self.W1,self.b1]) )) )
            #l1p_sy =  tf.scalar_summary( "L1_penalty" , l1_penalty )
            tot_loss = l2_loss #+ self.ALPHA * l1_penalty
            tot_loss_sy =  tf.scalar_summary( "loss" , tot_loss )
            
            _, y_var = tf.nn.moments(self.vars.y, [0,1])
            rsq =  1 - l2_loss / y_var
            rsq_sy = tf.scalar_summary( "R2", rsq)
            
        return tot_loss
###############################################
class ctflearn(tflearn):
    """
    **classification** template class for sklearn-style tensorflow wrapper
    """
    def _create_loss(self):
        """
        define loss variable and summaries;
        the method must return a tf.Variable (not a summary!)
        """
        raise NotImplementedError              
        def loss(logits, labels, NUM_CLASSES):
            # copied from tensorflow/tensorflow/examples/tutorials/mnist/mnist.py
            """Calculates the loss from the logits and the labels.
            Args:
            logits: Logits tensor, float - [batch_size, NUM_CLASSES].
            labels: Labels tensor, int32 - [batch_size].
            Returns:
            loss: Loss tensor of type float.
            """
            # Convert from sparse integer labels in the range [0, NUM_CLASSES)
            # to 1-hot dense float vectors (that is we will have batch_size vectors,
            # each with NUM_CLASSES values, all of which are 0.0 except there will
            # be a 1.0 in the entry corresponding to the label).
            batch_size = tf.size(labels)
            labels = tf.expand_dims(labels, 1)
            indices = tf.expand_dims(tf.range(0, batch_size), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense(
              concated, tf.pack([batch_size, NUM_CLASSES]), 1.0, 0.0)
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
                                                                  onehot_labels,
                                                                  name='xentropy')
            loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
            return loss
        NUM_CLASSES = self.vars.y.get_shape()[1]
        
        tot_loss = loss(logits, labels, NUM_CLASSES)
        return tot_loss
 
