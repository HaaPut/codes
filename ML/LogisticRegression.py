from __future__ import print_function

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
import timeit
import sys
import argparse


def load_data():
    X = np.random.rand(100,313)
    X[:50,:]  = X[:50,:] + 0.5
    y = np.zeros(100)
    y[:50] = 1
    #add 1 for bias term.
    X = np.hstack((np.ones((X.shape[0],1)), X))
    y = np.hstack(((y[:,np.newaxis],(1 - y[:,np.newaxis]))))
    return X,y

def split_train_test(X,y,split = 0.2,fold=1):
    """ split data into train/valid and test"""
    idx = int(split*y.shape[0])
    #to get same split every iteration or slice-picking set seed
    np.random.seed(421)
    randperm = np.random.permutation(y.shape[0])
    Xtrain = X.copy()
    ytrain = y.copy()
    Xtest = X[randperm[idx*(fold-1):idx*fold]]
    ytest = y[randperm[idx*(fold-1):idx*fold]]
    Xtrain = np.delete(Xtrain, range(idx*(fold-1),idx*fold),0)
    ytrain = np.delete(ytrain, range(idx*(fold-1),idx*fold),0)
    return Xtrain,Xtest,ytrain,ytest

def evaluate(y,y_):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def cost(logits, labels, w, beta = 0.01):
    regul = tf.nn.l2_loss(w,name='l2_norm')
    data = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=labels), name='mean_entropy')
    loss = tf.add(data, beta*regul,name='total_loss')
    tf.summary.scalar('WeightNorm',regul)
    tf.summary.scalar('cross_entropy',data)
    return loss

def kernel_logistic(beta = 0.1,d = 3,save_stuff=False):
    Xd, yd = load_data()
    input_dim = Xd.shape[1]
    learning_rate = 0.000001
    class_count = 2
    split = 0.2
    slices = int(1/split)
    iteration_count = 10000
    mod = 100
    N = int(Xd.shape[0]*(1 - split))

    #model setup
    X_  = tf.placeholder(tf.float64, shape=[None, input_dim],name='input_placeholder')
    Xtr  = tf.placeholder(tf.float64, shape=[N, input_dim],name='input_placeholder')
    y_ = tf.placeholder(tf.float64, shape=[None,class_count],name='label_placeholder')
    a = tf.Variable(tf.truncated_normal([class_count,N],dtype=tf.float64,stddev = 0.1),name='kernel_weight_vector')
    K = tf.add(tf.matmul(Xtr,X_,transpose_b=True) , 1)**d#compute kernel function values
    logits = tf.transpose(tf.matmul(a,K))
    y = tf.nn.softmax(logits)
    objective = cost(labels=y_, logits=logits, w=a, beta = beta)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(objective)
    
    accuracy = evaluate(y,y_)
    #tf.summary.scalar('Accuracy',accuracy)
    init = tf.global_variables_initializer()
    #merged = tf.summary.merge_all()
    
    training_xentropy = np.zeros((slices,iteration_count/mod))
    testing_xentropy = np.zeros((slices,iteration_count/mod))
    training_accuracy = np.zeros((slices,iteration_count/mod))
    testing_accuracy = np.zeros((slices,iteration_count/mod))
    with tf.Session() as sess:
        #train_writer = tf.summary.FileWriter('./logs' + '/train',sess.graph)
        for s in range(slices):
            k = 0
            print('running fold: ',s)
            Xtrain,Xtest,ytrain,ytest = split_train_test(Xd,yd,split = split,fold=s+1)
            sess.run(init)
            for i in range(iteration_count):
                _ = sess.run([optimizer],feed_dict={X_:Xtrain, y_:ytrain,Xtr:Xtrain})
                if (i+1)%mod == 0:
                    print('loss',objective.eval(session = sess,feed_dict={X_:Xtrain, y_:ytrain,Xtr:Xtrain}))
                if (i+1)%mod == 0 and save_stuff:
                    training_xentropy[s,k] = objective.eval(session = sess, feed_dict={X_:Xtrain,y_:ytrain,Xtr:Xtrain})
                    testing_xentropy[s,k] = objective.eval(session = sess, feed_dict = {X_:Xtest,y_:ytest,Xtr:Xtrain})
                    training_accuracy[s,k] = accuracy.eval(session=sess, feed_dict={X_:Xtrain,y_:ytrain,Xtr:Xtrain})
                    testing_accuracy[s,k] = accuracy.eval(session=sess, feed_dict={X_:Xtest,y_:ytest,Xtr:Xtrain})
                    k = k+1;
        if save_stuff:
            plt.figure("cross entropy")
            plt.plot(np.mean(training_xentropy,axis=0),'r',np.mean(testing_xentropy,axis=0),'b')
            plt.savefig('xentropy_'+str(d)+'_.png', bbox_inches='tight')
            plt.close()
            plt.figure("accuracy")
            plt.plot(np.mean(training_accuracy,axis=0),'r',np.mean(testing_accuracy,axis=0),'b')
            plt.savefig('accuracy_'+str(d)+'_.png', bbox_inches='tight')
            plt.close()

def logistic_regression(beta = 0.01,save_stuff=False):
    #read data
    Xd, yd = load_data()
    #parameters
    learning_rate = 0.01
    input_dim = Xd.shape[1]
    class_count = 2
    iteration_count = 5000
    mod = 100
    split = 0.2
    slices = int(1/split)

    #model setup
    X_  = tf.placeholder(tf.float64, shape=[None, input_dim],name='input_placeholder')
    y_ = tf.placeholder(tf.float64, shape=[None,class_count],name='label_placeholder')
    W = tf.Variable(tf.truncated_normal([input_dim,class_count],dtype=tf.float64,stddev=0.01),name='weight_vector')
    logits = tf.matmul(X_,W)
    y = tf.nn.softmax(logits)


    objective = cost(logits=logits, labels=y_, w=W, beta = beta)
    WNorm = tf.sqrt(2*tf.nn.l2_loss(W))
    
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(objective)
    init = tf.global_variables_initializer()
    #plots
    W_norm =  np.zeros((slices,iteration_count/mod))
    training_accuracy = np.zeros((slices,iteration_count/mod))
    test_accuracy =  np.zeros((slices,iteration_count/mod))
    training_loss =  np.zeros((slices,iteration_count/mod))
    test_loss =  np.zeros((slices,iteration_count/mod))

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    with tf.Session() as sess:
        for s in range(slices):
            k = 0
            sess.run(init)
            print('running fold',s  +1) 
            Xtrain,Xtest,ytrain,ytest = split_train_test(Xd,yd,split = split,fold=s+1)
            for i in range(iteration_count):
                _,loss,Wl2 = sess.run([optimizer,objective,WNorm],feed_dict={X_:Xtrain, y_:ytrain})
                if (i+ 1)%500 == 0:
                    print('loss',loss)
                if (i+1)%mod == 0 and save_stuff == True:
                    W_norm[s,k] = Wl2
                    training_accuracy[s,k] = (accuracy.eval({X_:Xtrain, y_:ytrain}))
                    test_accuracy[s,k] = (accuracy.eval({X_:Xtest, y_:ytest}))
                    training_loss[s,k] = (loss)
                    test_loss[s,k] = (objective.eval({X_:Xtest,y_:ytest}))
                    k = k+1
        if save_stuff:
            plt.figure()
            plt.plot(np.mean(W_norm,axis=0),'g')
            plt.savefig('W_norm_b_'+str(beta)+'_LR.png');plt.close()
            plt.figure()
            plt.plot(np.mean(training_accuracy,axis=0),'r', np.mean(test_accuracy,axis=0),'b')
            plt.savefig('accuracy_b_'+str(beta)+'_LR_.png');plt.close()
            plt.figure()
            plt.plot(np.mean(training_loss,axis=0),'r', np.mean(test_loss,axis=0),'b')
            plt.savefig('xentropy_b_'+str(beta)+'_LR_.png');plt.close()


def main(_):
    start_time = timeit.default_timer()
    kernel_logistic(d=2)
    print('Time taken by Kernel regression = ',timeit.default_timer()-start_time)

    #start_time = timeit.default_timer()
    #logistic_regression()
    #print('Time taken by Logistic regression = ',timeit.default_timer()-start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
