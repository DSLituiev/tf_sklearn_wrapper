a `scikit-learn` style wrapper around tensorflow package

usage (see `tf_lasso.py`):

    import tensorflow as tf
    from tflearn import tflearn, vardict

    class tflasso(tflearn):
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

    tflo = tflasso(ALPHA = 2e-1, checkpoint_dir = "./lasso-chpt/", dropout = None)
    tflo.fit( train_X, train_Y , load = True)
    tflo.tranform(test_X, test_Y)
