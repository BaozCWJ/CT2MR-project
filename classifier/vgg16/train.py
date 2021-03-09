import numpy as np
import tensorflow as tf
import os
import vgg16
import utils

img1 = utils.load_image("./test_data/tiger.jpeg")
img2 = utils.load_image("./test_data/puzzle.jpeg")

# batch1 = img1.reshape((1, 224, 224, 3))
# batch2 = img2.reshape((1, 224, 224, 3))

batch1 = np.ones([1,24,24,3])
batch2 = np.ones([1,24,24,3])

batch_images = np.concatenate((batch1, batch2), 0)
batch_labels = np.asarray([1,0])

max_iterations=5000

vgg = vgg16.Vgg16()
images = tf.placeholder(tf.float32, [None, 24,24,3])
labels = tf.placeholder(tf.int32, [None])

probs,logits = vgg.build(images)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
train_step = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), labels)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver=tf.train.Saver()

with tf.device('/cpu:0'):
    with tf.Session() as sess:
        #
        sess.run(tf.global_variables_initializer())
        for itr in range(1,max_iterations+1):
            #batch_images,batch_labels=batch_generator()
            feed_dict = {labels: batch_labels,images:batch_images}
            _,err,ac = sess.run([train_step,loss,acc],feed_dict=feed_dict)
            print(sess.run(probs,feed_dict={images:batch_images}))
            print(err,ac)

            if itr % 500 == 0:
                saver.save(sess, './model'+str(itr)+'.ckpt')
