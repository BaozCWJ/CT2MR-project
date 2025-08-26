import tensorflow as tf
import numpy as np
from resnet50 import ResNet50
from vgg16 import Vgg16
from data import read_image,minibatches,index4_4
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
    train_images,train_labels = read_image(path+'/train',img_type)
    val_images,val_labels = read_image(path+'/val',img_type)
    test_images,test_labels = read_image(path+'/test',img_type)

    img_type=img_type.replace('/','')

    Y_hat, model_params = ResNet50(input_shape=[256, 256, 1], classes=2)
    #Y_hat = tf.sigmoid(Z)

    images = model_params['input']
    labels = tf.placeholder(dtype=tf.int32, shape=[None, 1])

    logits = model_params['out']['Z']
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    prediction = tf.cast(tf.argmax(logits,1),tf.int32)

    saver = tf.train.Saver()

    loss_log = {'train':[],'val':[],'test':[]}
    index_log = {'train':[],'val':[],'test':[]}

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        for epoch in range(1,max_epochs+1):
            start_time = time.time()
            for batch_images,batch_labels in minibatches(train_images, train_labels, batch_size, shuffle=True):

                feed_dict = {labels: batch_labels,images:batch_images}
                _,err= sess.run([train_step,loss],feed_dict=feed_dict)

            err1,pred = sess.run([loss,prediction],feed_dict={labels: train_labels,images:train_images})
            index1 = index4_4(train_labels,pred)
            end_time=time.time()
            print('epoch:'+str(epoch)+' train_time:'+str(end_time-start_time)+' train_loss:'+str(err1))
            print('train_index: acc\tprec\trec\tF1')
            print(index1[4:])

            #validation
            err2,pred = sess.run([loss,prediction],feed_dict={labels: val_labels,images:val_images})
            index2 = index4_4(val_labels,pred)
            end_time=time.time()
            print('epoch:'+str(epoch)+' val_time:'+str(end_time-start_time)+' val_loss:'+str(err2))
            print('val_index: acc\tprec\trec\tF1')
            print(index2[4:])

            #test
            err3,pred = sess.run([loss,prediction],feed_dict={labels: test_labels,images:test_images})
            index3 = index4_4(test_labels,pred)
            end_time=time.time()
            print('epoch:'+str(epoch)+' test_time:'+str(end_time-start_time)+' test_loss:'+str(err3))
            print('test_index: acc\tprec\trec\tF1')
            print(index3[4:])

            loss_log['train'].append(err1)
            loss_log['val'].append(err2)
            loss_log['test'].append(err3)

            index_log['train'].append(index1)
            index_log['val'].append(index2)
            index_log['test'].append(index3)

            if epoch % 500 == 0:
                saver.save(sess, './model/resnet_'+img_type+str(epoch)+'.ckpt')

    np.save('resnet_'+img_type+'_loss.npy',loss_log)
    np.save('resnet_'+img_type+'_index.npy',index_log)

def train_vgg(img_type):
    #data
    train_images,train_labels = read_image(path+'/train',img_type)
    val_images,val_labels = read_image(path+'/val',img_type)
    test_images,test_labels = read_image(path+'/test',img_type)

    img_type=img_type.replace('/','')
    #model
    vgg = Vgg16()
    images = tf.placeholder(tf.float32, [None, 256,256,1])
    labels = tf.placeholder(tf.int32, [None,1])

    probs,logits = vgg.build(images)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
    prediction = tf.cast(tf.argmax(logits,1),tf.int32)

    saver=tf.train.Saver()

    loss_log = {'train':[],'val':[],'test':[]}
    index_log = {'train':[],'val':[],'test':[]}

    #training part
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1,max_epochs+1):
            start_time = time.time()
            #training
            for batch_images,batch_labels in minibatches(train_images, train_labels, batch_size, shuffle=True):

                feed_dict = {labels: batch_labels,images:batch_images}
                _,err= sess.run([train_step,loss],feed_dict=feed_dict)

            err1,pred = sess.run([loss,prediction],feed_dict={labels: train_labels,images:train_images})
            index1 = index4_4(train_labels,pred)
            end_time=time.time()
            print('epoch:'+str(epoch)+' train_time:'+str(end_time-start_time)+' train_loss:'+str(err1))
            print('train_index: acc\tprec\trec\tF1')
            print(index1[4:])

            #validation
            err2,pred = sess.run([loss,prediction],feed_dict={labels: val_labels,images:val_images})
            index2 = index4_4(val_labels,pred)
            end_time=time.time()
            print('epoch:'+str(epoch)+' val_time:'+str(end_time-start_time)+' val_loss:'+str(err2))
            print('val_index: acc\tprec\trec\tF1')
            print(index2[4:])

            #test
            err3,pred = sess.run([loss,prediction],feed_dict={labels: test_labels,images:test_images})
            index3 = index4_4(test_labels,pred)
            end_time=time.time()
            print('epoch:'+str(epoch)+' test_time:'+str(end_time-start_time)+' test_loss:'+str(err3))
            print('test_index: acc\tprec\trec\tF1')
            print(index3[4:])

            loss_log['train'].append(err1)
            loss_log['val'].append(err2)
            loss_log['test'].append(err3)

            index_log['train'].append(index1)
            index_log['val'].append(index2)
            index_log['test'].append(index3)

            if epoch % 500 == 0:
                saver.save(sess, './model/vgg_'+img_type+str(epoch)+'.ckpt')

    np.save('vgg_'+img_type+'_loss.npy',loss_log)
    np.save('vgg_'+img_type+'_index.npy',index_log)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.model=='resnet':
        train_resnet(img_type = args.img_type)
    else:
        train_vgg(img_type = args.img_type)
