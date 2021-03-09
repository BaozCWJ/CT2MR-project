import tensorflow as tf
import numpy as np
from model import ResNet50
from datalab import DataLabTrain, DataLabTest
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def train():
    Y_hat, model_params = ResNet50()
    #Y_hat = tf.sigmoid(Z)

    X = model_params['input']
    Y_true = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    Z = model_params['out']['Z']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y_true))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(Y_hat,1),tf.int32), Y_true)
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        try:
            train_gen = DataLabTrain('./datasets/train_set/').generator()
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            itr = 0
            for X_true, Y_true_ in train_gen:
                itr += 1
                if itr % 10 == 0:
                    l, ac,_ = sess.run([loss,acc, train_step], feed_dict={X:X_true, Y_true:Y_true_})
                    #acc = np.mean(y.astype('int32') == Y_true_.astype('int32'))
                    print('epoch: ' + str(itr) + ' loss: ' + str(l)+' accu: '+str(ac))
                else:
                    sess.run([train_step], feed_dict={X: X_true, Y_true: Y_true_})

                if itr % 500 == 0:
                    saver.save(sess, path + './model'+str(itr)+'.ckpt')

                if itr == 5000:
                    break
        finally:
            sess.close()


if __name__ == '__main__':
    train()
