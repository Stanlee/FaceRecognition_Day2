import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt


def prelu(input):

    alpha = tf.get_variable('prelu_alpha', input.get_shape()[-1], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
    output = tf.maximum(0.0, input) + alpha * tf.minimum(0.0, input)

    return output

def get_center_loss(features, labels, num_classes):

    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    labels = tf.reshape(labels, [-1])
    centers_batch = tf.gather(centers, labels) # feature에 해당하는 클래스를 센터와 맞춰서 배치로 만든다.

    # TODO: center loss
    loss = tf.nn.l2_loss(features-centers_batch)

    return loss


def get_triplet_loss(anchor, positive, negative, alpha):

    # TODO: triplet loss
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)),1) # 1, summarise in feature directions
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)),1) #

    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0),0) # to secure robustness for the changeable batch size
    # accuracy can be increased using dynamic learning rate for each batch size
    return loss


def get_triplet(dataset, n_labels, n_triplets):

    def get_one_triplet(input_data, input_labels):
        # Getting a pair of clients
        index = np.random.choice(n_labels, 2, replace=False)
        label_positive = index[0]
        label_negative = index[1]

        # Getting the indexes of the data from a particular client
        indexes = np.where(input_labels == index[0])[0]
        np.random.shuffle(indexes)

        # Picking a positive pair
        data_anchor = input_data[indexes[0], :, :, :]
        data_positive = input_data[indexes[1], :, :, :]

        # Picking a negative sample
        indexes = np.where(input_labels == index[1])[0]
        np.random.shuffle(indexes)
        data_negative = input_data[indexes[0], :, :, :]

        return data_anchor, data_positive, data_negative, label_positive, label_positive, label_negative

    target_data = dataset.train.images
    target_labels = dataset.train.labels
    c = target_data.shape[3]
    w = target_data.shape[1]
    h = target_data.shape[2]

    data_apn = np.zeros(shape=(n_triplets*3, w, h, c), dtype='float32')
    labels_apn = np.zeros(shape=n_triplets*3, dtype='uint64')
    for i in range(n_triplets):
        a, p, n, al, pl, nl = get_one_triplet(target_data, target_labels)
        data_apn[i*3,:,:,:]= a
        data_apn[i*3+1,:,:,:] = p
        data_apn[i*3+2,:,:,:] = n
        labels_apn[i*3]= al
        labels_apn[i*3+1] = pl
        labels_apn[i*3+2] = nl

    return data_apn, labels_apn

# conv, conv, maxpooling
def inference(input_images):

    net = slim.conv2d(input_images, num_outputs=32, kernel_size=3, padding='SAME', scope='conv1a')
    net = slim.conv2d(net, num_outputs=32, kernel_size=3, padding='SAME', scope='conv1b')
    net = slim.max_pool2d(net, kernel_size=2, scope='pool1')

    net = slim.conv2d(net, num_outputs=64, kernel_size=3, padding='SAME', scope='conv2a')
    net = slim.conv2d(net, num_outputs=64, kernel_size=3, padding='SAME', scope='conv2b')
    net = slim.max_pool2d(net, kernel_size=2, scope='pool2')

    net = slim.conv2d(net, num_outputs=128, kernel_size=3, padding='SAME', scope='conv3a')
    net = slim.conv2d(net, num_outputs=128, kernel_size=3, padding='SAME', scope='conv3b')
    net = slim.max_pool2d(net, kernel_size=2, scope='pool3')

    net = slim.flatten(net)

    feature = slim.fully_connected(net, num_outputs=2, activation_fn=None, scope='fc1')
    net = prelu(feature)

    net = slim.fully_connected(net, num_outputs=10, activation_fn=None, scope='fc2')

    return net, feature


def build_network(input_images, labels, num_classes, mode, center_loss_ratio=0.5, triplet_loss_alpha=0.2):

    logits, features = inference(input_images)

    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss = get_center_loss(features, labels, num_classes)
        with tf.name_scope('triplet_loss'):
            embeddings = tf.nn.l2_normalize(features, 1, 1e-10, name='embeddings')
            anchor, positive, negative = tf.unstack(tf.reshape(embeddings, [-1, 3, 2]), 3, 1) # anchor, positive, positive stack excerpt
            triplet_loss = get_triplet_loss(anchor, positive, negative, triplet_loss_alpha)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            if mode == 'Center loss':
                total_loss = softmax_loss + center_loss_ratio * center_loss
            elif mode == 'Triplet loss':
                total_loss = triplet_loss
            else:
                total_loss = softmax_loss

    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))

    return logits, features, embeddings, total_loss, accuracy


# Parameters
num_classes = 10 # fixed m_nist
mode = 'Triplet loss' # Center loss, Triplet loss, Softmax loss
training_epoch = 20
batch_size = 200
is_display = True #save the data for each epoch

os.environ['CUDA_VISIBLE_DEVICES'] = '0' # showing device information when executing

mnist = input_data.read_data_sets('/tmp/mnist', reshape=False)
mean_data = np.mean(mnist.train.images, axis=0)  #
total_batch = int(mnist.train.num_examples/batch_size)


with tf.name_scope('input'):
    # TODO: create placeholder (input_images, labels) // MNIST images : tf.float32, 28x28x1 // MNIST labels : tf.int64
    # iput_images
    input_images = tf.placeholder(tf.float32, shape=(None, 28,28,1), name='input_images')
    # labels =
    labels = tf.placeholder(tf.int64, shape=(None), name='global_step')

# TODO: create global step variable
# global_step =  to save the check point for memorizing the step
global_step = tf.Variable(0, trainable=False, name='global_step')

logits, features, embeddings, total_loss, accuracy = build_network(input_images, labels, num_classes, mode)

# TODO: optimaizer, adam
# optimizer =
optimizer = tf.train.AdamOptimizer(0.001)

# TODO: optimizer minimize
# train_op =
train_op = optimizer.minimize(total_loss, global_step=global_step)

with tf.Session() as sess:
    # TODO: global variable initializer
    # sess.run( )
    sess.run(tf.global_variables_initializer())

    # TODO: session run global step
    # step =
    step = sess.run(global_step)

    for epoch in range(training_epoch):
        epoch_loss = []
        epoch_acc = []

        for s in range(total_batch):
            if mode == 'Triplet loss':
                batch_images, batch_labels = get_triplet(mnist, num_classes, batch_size) #to make sets
            else:
                batch_images, batch_labels = mnist.train.next_batch(batch_size)

            if batch_images.size > 0:
                feed_dict = {input_images: batch_images - mean_data, labels: batch_labels}
                # TODO: session run (train_op, total_loss, accuracy)
                # _, loss, train_acc = sess.run( )
                _, loss, train_acc = sess.run(
                    [train_op, total_loss, accuracy],
                    feed_dict={input_images: batch_images - mean_data, labels: batch_labels}
                )

            epoch_loss.append(loss)
            epoch_acc.append(train_acc)
            step += 1

        print("Epoch:%d, loss:%.4f" % (epoch+1, np.mean(epoch_loss)))
        if mode != 'Triplet loss':
            val_image = mnist.validation.images - mean_data
            val_acc = sess.run(accuracy, feed_dict={input_images: val_image, labels: mnist.validation.labels})
            print("train_acc:%.4f, val_acc:%.4f" % (np.mean(epoch_acc), val_acc))

        if is_display:
            if mode != 'Triplet loss':
                feat = sess.run(features, feed_dict={input_images: mnist.train.images[:5000] - mean_data})
            else:
                feat = sess.run(embeddings, feed_dict={input_images: mnist.train.images[:5000] - mean_data})
            lab = mnist.train.labels[:5000]

            f = plt.figure(figsize=(16, 9))
            c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
                 '#ff00ff', '#990000', '#999900', '#009900', '#009999']
            for i in range(10):
                plt.plot(feat[lab == i, 0].flatten(), feat[lab == i, 1].flatten(), '.', c=c[i])
            plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            plt.grid()
            plt.title('Train (' + mode + ') epoch : ' + str(epoch+1))
            f.savefig('train_' + str(epoch+1) + '.png')
            plt.close(f)

    if is_display:
        if mode != 'Triplet loss':
            feat = sess.run(features, feed_dict={input_images: mnist.test.images[:5000] - mean_data})
        else:
            feat = sess.run(embeddings, feed_dict={input_images: mnist.test.images[:5000] - mean_data})
        lab = mnist.test.labels[:5000]

        f = plt.figure(figsize=(16, 9))
        c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
             '#ff00ff', '#990000', '#999900', '#009900', '#009999']
        for i in range(10):
            plt.plot(feat[lab==i,0].flatten(), feat[lab==i,1].flatten(), '.', c=c[i])
        plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        plt.grid()
        plt.title('Test (' + mode + ')')
        f.savefig('test_' + str(epoch+1) + '.png')
        plt.show()
