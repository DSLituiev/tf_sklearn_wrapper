#!/usr/bin/env python3
# coding: utf-8

from __future__ import print_function
import  sys, os, gzip, pickle
import urllib.request, urllib.parse, urllib.error
import logging
from tqdm import tqdm

logger = logging.StreamHandler()
logger.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
logger.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(logger)

info = lambda x: print(x , file = sys.stderr)

import numpy as np
from tflearn import ctflearn, vardict, batchgen
import tensorflow as tf

class tfgan(ctflearn):
    """ Generative Adversarial Network implementation 
        Reference:
        http://arxiv.org/pdf/1406.2661.pdf
    """
    def _create_network_g(self):
        if "ng1"  not in self.__dict__:
            self.ng1 = 10
        if "zlen"  not in self.__dict__:
            self.zlen = 10
  
        """Generative model"""
        """this is a random vector; 
        logp ~ - x.T @ (sigma^2 * eye(p)) @ x """        
        #if not hasattr(self, "vars") or len(self.vars.keys()) > 0:
        info( " creating gen model ; %s" % repr(self.zlen) )
        #self.vars["z"] = tf.placeholder("float", shape=[None, self.zlen])
        self.vars["z"] = tf.placeholder("float", shape=[None, self.zlen])
        "Variables"        
        print( " creating gen model ")
        with tf.variable_scope('generative') as bigscope:
            with tf.variable_scope('dense1') as scope:
                self.parameters["g1W1"] = tf.Variable(tf.truncated_normal([self.ng1, self.zlen], stddev=0.1), name="weight")
                self.parameters["g1b1"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
            
            with tf.variable_scope('dense2') as scope:
                self.parameters["g2W1"] = tf.Variable(tf.truncated_normal([self.xlen, self.ng1], stddev=0.1), name="weight")
                self.parameters["g2b1"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
            
        "Graph"
        self.vars["g1out"] = tf.sigmoid( tf.matmul( self.vars.z, tf.transpose(self.g1W1)) + self.g1b1 )
        self.vars["x_predicted"] = tf.sigmoid( tf.matmul( self.vars.g1out, tf.transpose(self.g2W1)) + self.g2b1 )
        
        self.saver = tf.train.Saver()

        info( "var keys: %s " %  repr(self.vars.keys() ) )
        return self.vars.x_predicted

    def _create_network_d(self, x_gen = None):
        info( " creating discr model ")
        if "nd1"  not in self.__dict__:
            self.nd1 = 10
        #if not hasattr(self, "vars") or len(self.vars.keys()) > 0:
        #    self.vars = vardict()
        if x_gen is None:
            self.vars["x"] = tf.placeholder("float", shape=[None, self.xlen])
        else:
            self.vars["x"] = x_gen
        self.vars["y"] = tf.placeholder("float", shape=[None, 1])

        """ Discriminative Model """
        "Variables"
        with tf.variable_scope('discriminative') as bigscope:
            with tf.variable_scope('dense1') as scope:
                self.parameters["d1W1"] = tf.Variable(tf.truncated_normal([self.nd1, self.xlen], stddev=0.1), name="weight")
                self.parameters["d1b1"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")

            with tf.variable_scope('dense2') as scope:
                self.parameters["d2W1"] = tf.Variable(tf.truncated_normal([1, self.nd1], stddev=0.1), name="weight")
                self.parameters["d2b1"] = tf.Variable(tf.constant(0.1, shape=[1, 1]), name="bias")
        "Graph"
        self.vars["d1out"] = tf.sigmoid( tf.matmul( self.vars.x, tf.transpose(self.d1W1)) + self.d1b1 )
        self.vars["y_predicted"] = tf.sigmoid( tf.matmul( self.vars.d1out, tf.transpose(self.d2W1)) + self.d2b1 )
            
        self.saver = tf.train.Saver()
        info( "var keys: %s " %  repr(self.vars.keys() ) )
        return self.vars.y_predicted
        
    def _create_loss(self):
        """
        define loss variable and summaries;
        the method must return a tf.Variable (not a summary!)
        """
        with tf.name_scope("loss") as scope:
            tot_loss = tf.nn.sigmoid_cross_entropy_with_logits(self.vars.y_predicted , self.vars.y)
        return tot_loss

    def _generate_(self, sess, feed_dict, load = False):
        #print("feed_dict=",feed_dict, file = sys.stderr)
        predicted = sess.run( self.vars.x_predicted,
                        feed_dict = feed_dict )
        return predicted

    ##############################################3
    def get_generative_input(self):
        """ compose a generative input set """
        Z = np.random.randn( self.g_samples, self.zlen )
        "random noise"
        g_feed_dict = { self.vars.z: Z, }
        if self.dropout > 0 and self.dropout < 1:
            g_feed_dict[ self.vars.keep_prob] = self.dropout
        return g_feed_dict
    ##############################################3
    def fit(self, train_X, train_Y , test_X= None, test_Y = None, load = True,
                 epochs = 10,
                 d_epochs = 2, g_epochs = 1):
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
            self.vars = vardict()
            x_gen = self._create_network_g( )
            y_predicted = self._create_network_d( x_gen )
            # info( "var keys: %s " %  repr(self.vars.keys() ) )

            """ must be called from within a graph scope """
            sess_config = tf.ConfigProto(inter_op_parallelism_threads=self.NUM_CORES,
                                       intra_op_parallelism_threads= self.NUM_CORES)
            # Initializing the variables
            init = tf.initialize_all_variables()
            with  tf.Session(config = sess_config) as sess:
                if load:
                    self._load_(sess)
                else:
                    sess.run(init)
                """ x_gen == self.vars.x_predicddted, """
                if not ("keep_prob" in self.vars or hasattr( self.vars, "keep_prob") ):
                    self.dropout = 0.0

                """ pre-DISCRIMINATIVE  phase """
                #info( "var keys: %s " %  repr(self.vars.keys() ) )
                g_feed_dict =  self.get_generative_input()

                info(" generating samples ... ",)
                X_predicted = self._generate_(sess, g_feed_dict,)
                """ DISCRIMINATIVE  phase """
                """ compose a discriminative training set """
                Xmix = np.vstack( ( train_X, X_predicted ) )
                Ymix = np.vstack( (np.ones( (train_X.shape[0], 1)) , np.zeros( (X_predicted.shape[0], 1) ) ) )

                np.random.shuffle( Xmix )
                np.random.shuffle( Ymix )
                #_y_ = np.reshape(_y_, [-1, 1])
                d_feed_dict = { self.vars.x : Xmix, self.vars.y : Ymix}

                """ Discriminative  training"""
                discriminative_tot_loss = self._create_loss()
                d_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminative" )
                d_train_op = self.optimizer( self.learning_rate ).minimize( discriminative_tot_loss , var_list= d_train_vars)
                
                info("training the discriminative model ... ",)
                self._train_submodel(sess, d_train_op, d_feed_dict, epochs = d_epochs)

                """ GENERATIVE  phase """
                """ compose a generative input set """
                g_feed_dict =  self.get_generative_input()
                """ Generative training """
                "! loss is negative "
                g_total_loss = - self._create_loss() 
                g_train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generative")
                g_train_op = self.optimizer( self.learning_rate ).minimize( g_total_loss , var_list= g_train_vars)

                info("training the generative model ... ",)
                self._train_submodel( sess, g_train_op, g_feed_dict, epochs = g_epochs)
        return

    def _train_submodel(self, sess, train_op , train_feed_dict,
                                test_feed_dict = None,
                                load = True,  epochs = 8 ):

            # Merge all the summaries and write them out
            summary_op = tf.merge_all_summaries()

            # Initializing the variables
            init = tf.initialize_all_variables()
            " training per se"
            getb = batchgen( self.BATCH_SIZE, dictionary = True)

            # write summaries out
            summary_writer = tf.train.SummaryWriter("./tmp/mnist_logs", sess.graph_def)
            summary_proto = tf.Summary()
            # Fit all training data
            print("training epochs: %u ... %u, saving each %u' epoch" % \
                    (self.last_ckpt_num, self.last_ckpt_num + epochs, self.display_step),
                    file = sys.stderr)
            for macro_epoch in tqdm(range( self.last_ckpt_num//self.display_step ,
                                     (self.last_ckpt_num + epochs)//  self.display_step )):
                "do minibatches"
                for subepoch in tqdm(range(self.display_step)):
                    for batch_train_dict in getb( train_feed_dict ):

                        if self.dropout:
                            feed_dict[self.vars.keep_prob] =  self.dropout
                        sess.run(train_op, feed_dict = feed_dict)
                epoch = macro_epoch * self.display_step

                """ Display logs once in `display_step` epochs """
                _sets_ = {"train" :  train_feed_dict }
                summaries = {}
                summaries_plainstr = []
                if (test_feed_dict is not None):
                    if all( (type(x) is str for x  in test_feed_dict.keys()) ):
                        _sets_[ "test" ] = { self.vars[ kk ] : vv for kk, vv  in test_feed_dict }

                for _set_, feed_dict in  _sets_.items():
                    if self.dropout:
                        feed_dict[ self.vars.keep_prob ] = self.dropout

                    summary_str = sess.run(summary_op, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, epoch)
                    summary_d = summary_dict(summary_str, summary_proto)
                    summaries[_set_] = summary_d
                    #summary_d["epoch"] = epoch

                    summaries_plainstr.append(  "\t".join(["",_set_] +["{:s}: {:.4f}".format(k,v) if type(v) is float else "{:s}: {:s}".format(k,v) for k,v in summary_d.items() ]) )

                    self.train_summary.append( summaries["train"] )
                    if  "test" in summaries:
                        self.test_summary.append( summaries["test"] )

                    logstr = "Epoch: {:4d}\t".format(epoch) + "\n"+ "\n".join(summaries_plainstr)
                    print(logstr, file = sys.stderr )
                    self.saver.save(sess, self.checkpoint_dir + '/' +'model.ckpt',
                       global_step=  epoch)
                    self.last_ckpt_num = epoch
                print("Optimization Finished!", file = sys.stderr)



        
if __name__ == "__main__":

    fname = 'mnist/mnist.pkl.gz'
    if not os.path.isfile(fname):
        info("downloading MNIST")
        testfile = urllib.request.URLopener()
        testfile.retrieve("http://deeplearning.net/data/mnist/mnist.pkl.gz", fname)
    f = gzip.open(fname, 'rb')
    train_set, valid_set, test_set = pickle.load(f,  encoding='latin1')

    X_train, Y_train = train_set

    """ initialize the models """
    info("initializing the model")
    gan = tfgan( g_samples = 50, epochs = 50 )
    info("fitting the model")
    gan.fit( X_train, Y_train , load = False)
