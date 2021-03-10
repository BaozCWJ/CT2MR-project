import tensorflow as tf
import numpy as np
from resnet50 import ResNet50
from vgg16 import Vgg16
from data import read_image,minibatches
import os
import time
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--model",default="resnet", type=str, help='architecture of model, e.g. vgg or resnet')
parser.add_argument("--img_type", default="/ct/", type=str, help="the type of images, e.g. /ct/")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#Hyperparameter
max_epochs=5000
path='./dataset'
batch_size=32



def train_resnet(img_type):
    #data
    train_imgs,train_labels = read_image(path,img_type)

    Y_hat, model_params = ResNet50(input_shape=[256, 256, 1], classes=2)
    #Y_hat = tf.sigmoid(Z)

    X = model_params['input']
    Y_true = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    Z = model_params['out']['Z']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z, labels=Y_true))
    train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(Y_hat,1),tf.int32), Y_true)
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(1,max_epochs+1):
            start_time = time.time()
            for batch_images,batch_labels in minibatches(train_imgs, train_labels, batch_size, shuffle=True):

                if epoch % 10 == 0:
                    l, ac,_ = sess.run([loss,acc, train_step], feed_dict={X:batch_images, Y_true:batch_labels})
                    print('epoch: ' + str(epoch) + ' loss: ' + str(l)+' accu: '+str(ac))
                else:
                    sess.run([train_step], feed_dict={X: batch_images, Y_true: batch_labels})

                if epoch % 500 == 0:
                    saver.save(sess, path + './model'+str(epoch)+'.ckpt')
            end_time=time.time()
            print(end_time-start_time)

def train_vgg(img_type):
    #data
    train_imgs,train_labels = read_image(path,img_type)

    #model
    vgg = Vgg16()
    images = tf.placeholder(tf.float32, [None, 256,256,1])
    labels = tf.placeholder(tf.int32, [None,1])

    probs,logits = vgg.build(images)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), labels)
    acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver=tf.train.Saver()

    #training part
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1,max_epochs+1):
            start_time = time.time()
            #training
            for batch_images,batch_labels in minibatches(train_imgs, train_labels, batch_size, shuffle=True):

                feed_dict = {labels: batch_labels,images:batch_images}
                _,err,ac = sess.run([train_step,loss,acc],feed_dict=feed_dict)

            end_time=time.time()
            print(end_time-start_time)

            if epoch % 500 == 0:
                saver.save(sess, './model'+str(epoch)+'.ckpt')


if __name__ == '__main__':
    args = parser.parse_args()
    if args.model=='resnet':
        train_resnet(img_type = args.img_type)
    else:
        train_vgg(img_type = args.img_type)
