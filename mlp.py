""" Code from https://github.com/aymericdamien/TensorFlow-Examples/
"""
import tensorflow as tf
import numpy as np

class MLP:
    def __init__(self, train_x, train_y, save_path, learning_rate=0.001, training_epochs=15,
                 batch_size=100, display_step=1, hidden_1=256, hidden_2=256):

        # Data
        self.train_x = train_x
        self.train_y = train_y
        self.save_path = save_path
        self.current = 0 # location of head of current batch
        self.num_train = len(train_y)

        # Learning Parameters
        self.learning_rate = learning_rate
        self.training_epochs = training_epochs
        self.batch_size = batch_size
        self.display_step = display_step
        

        # Network Parameters
        n_hidden_1 = hidden_1 # 1st layer number of neurons
        n_hidden_2 = hidden_2 # 2nd layer number of neurons
        n_input = 784 # MNIST data input (img shape: 28*28)
        n_classes = 2 # MNIST total classes (0-9 digits)
        
        # tf Graph input
        self.X = tf.placeholder("float", [None, n_input])
        self.Y = tf.placeholder("float", [None, n_classes])
        
        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes])) 
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

        weights = self.weights
        biases = self.biases
        
        # Computational graph
        layer_1 = tf.add(tf.matmul(self.X, weights['h1']), biases['b1'])
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        predictions = tf.matmul(layer_2, weights['out']) + biases['out']
        self.predictions = predictions

        # Define loss and optimizer
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=self.Y))
        # self.loss_op = tf.losses.mean_squared_error(labels=self.Y, predictions=self.predictions)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        self.pred = tf.argmax(tf.nn.softmax(self.predictions), 1)
        self.acc  = tf.reduce_mean(tf.cast(tf.equal(self.pred, tf.argmax(self.Y, 1)), 'float'))
        
        
        
        
        # Initializer
        self.init = tf.global_variables_initializer()
        # Saver
        self.saver = tf.train.Saver()


    def train(self, restore=True, batch_size=None,training_epochs=None, display_step=None):
        if training_epochs is None:
            training_epochs = self.training_epochs
        if batch_size is None:
            batch_size = self.batch_size
        if display_step is None:
            display_step = self.display_step
                
        train_x = self.train_x
        train_y = self.train_y
        init = self.init
        X = self.X
        Y = self.Y
        train_op = self.train_op
        loss_op = self.loss_op

    
        
        with tf.Session() as sess:
            try:
                if not restore:
                    raise
                self.saver.restore(sess, self.save_path)

            except:
                print('initializing')
                sess.run(init)

            # Training cycle
            for epoch in range(training_epochs):
                avg_cost = 0.
                total_batch = int(len(train_x)/batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = self.next_batch(batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
                                                                    Y: batch_y})
                    # Compute average loss
                    avg_cost += c / total_batch
                    # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
            print("Optimization Finished!")

            self.saver.save(sess, self.save_path)

    def predict(self, x):
        X = self.X
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            pred = sess.run(self.pred, feed_dict={X:x})
        return pred

    def accuracy(self, x, y):
        X = self.X
        Y = self.Y
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_path)
            acc = sess.run(self.acc, feed_dict={X:x, Y:y})
        return acc
        
    
    
    def next_batch(self, batch_size):
        """Gets the next batch size.
        """
        assert batch_size < self.num_train

        if batch_size + self.current > self.num_train:
            self.current = 0

        batch_x = self.train_x[self.current: self.current + batch_size]
        batch_y = self.train_y[self.current: self.current + batch_size]

        self.current += batch_size

        return batch_x, batch_y
