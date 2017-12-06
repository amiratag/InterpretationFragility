import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import applications
from keras import backend as K
#from keras_squeezenet import SqueezeNet
import tensorflow as tf
from influence import Influence, plot_mnist
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import normalize

def retrain_inception_on_flowers(seed=22, verbose=True, new_training_data=False):

    # dimensions of our images.
    img_width, img_height = 299, 299

    nb_train_samples = 1000
    nb_validation_samples = 200
    epochs = 10
    batch_size = 20
    
    tf.set_random_seed(seed)
    np.random.seed(seed)

    train_shuffle_idx = np.random.permutation(nb_train_samples)
    train_data = np.load('data/flower_photos/bottleneck_features_train-inception.npy')[train_shuffle_idx]
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2)).reshape(-1,1)[train_shuffle_idx]

    valid_shuffle_idx = np.random.permutation(nb_validation_samples)
    validation_data = np.load('data/flower_photos/bottleneck_features_validation-inception.npy')[valid_shuffle_idx]
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2)).reshape(-1,1)[valid_shuffle_idx]
    
    if new_training_data:
        il = ImageLoader()
        train_data = np.zeros((nb_train_samples, 2048)) 
        for i in range(nb_train_samples):
            if (i%100==0):
                print("Regenerating training data:",i)
            pic, feature = il.load_candidate_image(train_shuffle_idx[i], img_folder='train', dataset='flowers', model='Inception')
            train_data[i] = feature

        validation_data = np.zeros((nb_validation_samples, 2048)) 
        for i in range(nb_validation_samples):
            if (i%100==0):
                print("Regenerating validation data:",i)
            pic, feature = il.load_candidate_image(valid_shuffle_idx[i], img_folder='validation', dataset='flowers', model='Inception')
            validation_data[i] = feature
            
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    g = tf.Graph()
    with g.as_default():
        tf.set_random_seed(seed)
        X = tf.placeholder(tf.float32, shape=(None, 2048))
        W_flat = tf.get_variable("all_weights",[2049])

        W = tf.reshape(W_flat[0:2048], [2048, 1])
        b = tf.reshape(W_flat[2048], [1])

        y =  tf.nn.sigmoid(tf.matmul(X, W) + b)
        y_ = tf.placeholder(tf.float32, [None, 1],name="y_")

        cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y)) #yes, I know this isn't cross entropy...
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct_prediction = tf.equal(tf.round(y), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    

        sess = tf.Session(graph=g)
        np.random.seed(seed)
        tf.set_random_seed(seed)
        sess.run(tf.global_variables_initializer())

        for j in range(epochs):
            for i in range(10):
                batch_xs = train_data[100*i:100*(i+1)]
                batch_ys = train_labels[100*i:100*(i+1)]
                _, acc, cp = sess.run([train_step, accuracy, correct_prediction], feed_dict={X: batch_xs, y_: batch_ys})
            if verbose:
                print('Epoch '+str(j) + ', Batch Acc=' + str(acc))
    
            print("Test Accuracy:",sess.run(accuracy, feed_dict={X: validation_data,y_: validation_labels}))    
    return g, sess, W_flat, cross_entropy, X, y, y_, train_data, train_labels, train_shuffle_idx, valid_shuffle_idx


def retrain_VGG_on_flowers(seed=22, verbose=True):
    global train_shuffle_idx, valid_shuffle_idx
    
    np.random.seed(seed)
    
    label_names = ['Cat','Dog']
    # dimensions of our images.
    img_width, img_height = 150, 150

    top_model_weights_path = 'vgg16_weights.h5'
    nb_train_samples = 1000
    nb_validation_samples = 200
    epochs = 10
    batch_size = 20
    
    train_shuffle_idx = np.random.permutation(nb_train_samples)
    train_data = np.load('data/flower_photos/bottleneck_features_train-vgg.npy').reshape(-1,4*4*512)[train_shuffle_idx]
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2)).reshape(-1,1)[train_shuffle_idx]

    valid_shuffle_idx = np.random.permutation(nb_validation_samples)
    validation_data = np.load('data/flower_photos/bottleneck_features_validation-vgg.npy').reshape(-1,4*4*512)[valid_shuffle_idx]
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2)).reshape(-1,1)[valid_shuffle_idx]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, 4*4*512))
    W_flat = tf.Variable(tf.zeros([train_data.shape[1]+1]),name="W_flat")
    W = tf.reshape(W_flat[:-1], [train_data.shape[1], 1])
    b = W_flat[-1]

    y =  tf.nn.sigmoid(tf.matmul(X, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 1],name="y_")

    cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y)) #yes, I know this isn't cross entropy...
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for j in range(epochs):
        for i in range(10):
            batch_xs = train_data[100*i:100*(i+1)]
            batch_ys = train_labels[100*i:100*(i+1)]
            _, acc, cp = sess.run([train_step, accuracy, correct_prediction], feed_dict={X: batch_xs, y_: batch_ys})
        if verbose:
            print('Epoch '+str(j) + ', Batch Acc=' + str(acc))
    print("Test Accuracy:",sess.run(accuracy, feed_dict={X: validation_data,y_: validation_labels}))
        
    return sess, W_flat, cross_entropy, X, y, y_, train_data, train_labels, train_shuffle_idx, valid_shuffle_idx

def train_squeeze_on_cats_dogs(seed=43, verbose=True):
    np.random.seed(seed)

    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 10
    batch_size = 16

    train_shuffle_idx = np.random.permutation(nb_train_samples)
    train_data = np.load('bottleneck_features_train-squeeze.npy').reshape(-1,13*13*512)[train_shuffle_idx]
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2)).reshape(-1,1)[train_shuffle_idx]

    valid_shuffle_idx = np.random.permutation(nb_validation_samples)
    validation_data = np.load('bottleneck_features_validation-squeeze.npy').reshape(-1,13*13*512)[valid_shuffle_idx]
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2)).reshape(-1,1)[valid_shuffle_idx]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, 13*13*512))
    W_flat = tf.get_variable("all_weights",[1029])

    W_conv = tf.reshape(W_flat[:1024],[1, 1, 512, 2])
    b_conv = tf.reshape(W_flat[1024:1026], [2])
    W = tf.reshape(W_flat[-3:-1], [2, 1])
    b = W_flat[-1]
    
    conv = tf.reshape(X, [-1,13,13,512])
    conv = tf.nn.conv2d(conv, W_conv, strides=[1,1,1,1], padding='VALID')
    conv = tf.nn.bias_add(conv, b_conv)
    conv = tf.nn.sigmoid(conv)
    out = tf.nn.avg_pool(conv, (1, 13, 13, 1), (1,1,1,1), padding='VALID')
    out = tf.reshape(out, [-1, 2])

    y =  tf.nn.sigmoid(tf.matmul(out, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 1],name="y_")

    cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y)) #yes, I know this isn't cross entropy...
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for j in range(epochs):
        for i in range(20):
            batch_xs = train_data[100*i:100*(i+1)]
            batch_ys = train_labels[100*i:100*(i+1)]
            _, acc, cp = sess.run([train_step, accuracy, cross_entropy], feed_dict={X: batch_xs, y_: batch_ys})
        if (j%20==0):
            if (verbose):
                print('Epoch '+str(j) + ', Batch Acc=' + str(acc))
                print('Loss '+str(cp))
                print("Test Accuracy:",sess.run(accuracy, feed_dict={X: validation_data,y_: validation_labels}))    
    
    print("Final Test Accuracy:",sess.run(accuracy, feed_dict={X: validation_data,y_: validation_labels}))    

    return sess, W_flat, cross_entropy, X, y, y_, train_data, train_labels, train_shuffle_idx, valid_shuffle_idx


def train_squeeze_on_flowers(seed=43, verbose=True):
    np.random.seed(seed)

    train_data_dir = 'data/flower_photos/train'
    validation_data_dir = 'data/flower_photos/validation'
    nb_train_samples = 1000
    nb_validation_samples = 200
    epochs = 240
    batch_size = 20

    train_shuffle_idx = np.random.permutation(nb_train_samples)
    train_data = np.load('data/flower_photos/bottleneck_features_train-squeeze.npy').reshape(-1,13*13*512)[train_shuffle_idx]
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2)).reshape(-1,1)[train_shuffle_idx]

    valid_shuffle_idx = np.random.permutation(nb_validation_samples)
    validation_data = np.load('data/flower_photos/bottleneck_features_validation-squeeze.npy').reshape(-1,13*13*512)[valid_shuffle_idx]
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2)).reshape(-1,1)[valid_shuffle_idx]

    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, shape=(None, 13*13*512))
    W_flat = tf.get_variable("all_weights",[1029])

    W_conv = tf.reshape(W_flat[:1024],[1, 1, 512, 2])
    b_conv = tf.reshape(W_flat[1024:1026], [2])
    W = tf.reshape(W_flat[-3:-1], [2, 1])
    b = W_flat[-1]
    
    conv = tf.reshape(X, [-1,13,13,512])
    conv = tf.nn.conv2d(conv, W_conv, strides=[1,1,1,1], padding='VALID')
    conv = tf.nn.bias_add(conv, b_conv)
    conv = tf.nn.sigmoid(conv)
    out = tf.nn.avg_pool(conv, (1, 13, 13, 1), (1,1,1,1), padding='VALID')
    out = tf.reshape(out, [-1, 2])

    y =  tf.nn.sigmoid(tf.matmul(out, W) + b)
    y_ = tf.placeholder(tf.float32, [None, 1],name="y_")

    cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y)) #yes, I know this isn't cross entropy...
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for j in range(epochs):
        for i in range(10):
            batch_xs = train_data[100*i:100*(i+1)]
            batch_ys = train_labels[100*i:100*(i+1)]
            _, acc, cp = sess.run([train_step, accuracy, cross_entropy], feed_dict={X: batch_xs, y_: batch_ys})
        if (j%20==0):
            if (verbose):
                print('Epoch '+str(j) + ', Batch Acc=' + str(acc))
                print('Loss '+str(cp))
                print("Test Accuracy:",sess.run(accuracy, feed_dict={X: validation_data,y_: validation_labels}))    
    
    print("Final Test Accuracy:",sess.run(accuracy, feed_dict={X: validation_data,y_: validation_labels}))    

    return sess, W_flat, cross_entropy, X, y, y_, train_data, train_labels, train_shuffle_idx, valid_shuffle_idx

def train_convnet_on_mnist_fashion(seed=43, verbose=True):
    mnist = input_data.read_data_sets('data/fashion',one_hot=False)

    X_train, y_train = mnist.train.next_batch(10000)
    idx = np.where(y_train>7)[0][:1500]
    X_train = X_train[idx]; y_train = y_train[idx].reshape(-1,1)-8

    X_test, y_test = mnist.test.next_batch(1000)
    idx = np.where(y_test>7)[0]
    X_test = X_test[idx]; y_test = y_test[idx].reshape(-1,1)-8    
    
    np.random.seed(seed)
    tf.reset_default_graph()

    def conv2d(x, W, b, strides=1):
        # Conv2D wrapper, with bias and relu activation
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.sigmoid(x)


    def maxpool2d(x, k=2):
        # MaxPool2D wrapper
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                              padding='SAME')


    # Placeholders
    n_w_conv = 5*5*16
    n_b_conv = 16
    n_mult = 14*14*16
    n_bias = 1

    lengths = [n_w_conv, n_b_conv, n_mult, n_bias]
    idx = np.cumsum(lengths)

    W_flat = tf.Variable(tf.zeros([idx[-1]]),name="W_flat") #total number of weights

    i = 0
    W_conv = tf.reshape(W_flat[0:idx[i]],[5,5,1,16]);
    b_conv = tf.reshape(W_flat[idx[i]:idx[i+1]],[16]); i += 1
    W_fc = tf.reshape(W_flat[idx[i]:idx[i+1]],[14*14*16, 1]); i += 1
    b_fc = tf.reshape(W_flat[idx[i]:],[1]);

    x = tf.placeholder(tf.float32, [None, X_train.shape[1]],name="x")
    conv1a = tf.reshape(x, shape=[-1, 28, 28, 1])
    conv1b = conv2d(conv1a, W_conv, b_conv)
    conv1c = maxpool2d(conv1b, k=2)
    conv1d = tf.reshape(conv1c, [-1, n_mult])
    y =  tf.nn.sigmoid(tf.matmul(conv1d, W_fc) + b_fc)

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 1],name="y_")
    cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y)) #yes, I know this isn't cross entropy...
    train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

    correct_prediction = tf.equal(tf.round(y), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    # Train
    for _ in range(10):
        for i in range(15):
            batch_xs = X_train[100*i:100*(i+1)]
            batch_ys = y_train[100*i:100*(i+1)]

            _, acc = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys})
            print('\r',end=str(i)+' '+str(acc)+'  ')

    batch_xs, batch_ys = mnist.test.next_batch(1000)
    print("\nTest Accuracy:",sess.run(accuracy, feed_dict={x: X_test,y_: y_test}))
    
    return sess, W_flat, cross_entropy, x, y, y_, X_train, y_train

def retrain_VGG_on_cats_dogs(seed=22, verbose=True):
    global train_shuffle_idx, valid_shuffle_idx
    
    np.random.seed(seed)
    
    label_names = ['Cat','Dog']
    # dimensions of our images.
    img_width, img_height = 150, 150

    top_model_weights_path = 'vgg16_weights.h5'
    train_data_dir = 'data/train'
    validation_data_dir = 'data/validation'
    nb_train_samples = 2000
    nb_validation_samples = 800
    epochs = 10
    batch_size = 16
    
    train_shuffle_idx = np.random.permutation(nb_train_samples)
    train_data = np.load('bottleneck_features_train.npy').reshape(-1,4*4*512)[train_shuffle_idx]
    train_labels = np.array([0] * int(nb_train_samples / 2) + [1] * int(nb_train_samples / 2)).reshape(-1,1)[train_shuffle_idx]

    valid_shuffle_idx = np.random.permutation(nb_validation_samples)
    validation_data = np.load('bottleneck_features_validation.npy').reshape(-1,4*4*512)[valid_shuffle_idx]
    validation_labels = np.array([0] * int(nb_validation_samples / 2) + [1] * int(nb_validation_samples / 2)).reshape(-1,1)[valid_shuffle_idx]

    tf.reset_default_graph()
    g = tf.Graph()
    with g.as_default():

        X = tf.placeholder(tf.float32, shape=(None, 4*4*512))
        W_flat = tf.Variable(tf.zeros([train_data.shape[1]+1]),name="W_flat")
        W = tf.reshape(W_flat[:-1], [train_data.shape[1], 1])
        b = W_flat[-1]

        y =  tf.nn.sigmoid(tf.matmul(X, W) + b)
        y_ = tf.placeholder(tf.float32, [None, 1],name="y_")

        cross_entropy = tf.reduce_sum(tf.squared_difference(y_, y)) #yes, I know this isn't cross entropy...
        train_step = tf.train.AdamOptimizer().minimize(cross_entropy)

        correct_prediction = tf.equal(tf.round(y), y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        for j in range(epochs):
            for i in range(20):
                batch_xs = train_data[100*i:100*(i+1)]
                batch_ys = train_labels[100*i:100*(i+1)]
                _, acc, cp = sess.run([train_step, accuracy, correct_prediction], feed_dict={X: batch_xs, y_: batch_ys})
            if verbose:
                print('Epoch '+str(j) + ', Batch Acc=' + str(acc))
        print("Test Accuracy:",sess.run(accuracy, feed_dict={X: validation_data,y_: validation_labels}))
        
    return g, sess, W_flat, cross_entropy, X, y, y_, train_data, train_labels, train_shuffle_idx, valid_shuffle_idx

class ImageLoader():    
    def __init__(self, model_name='Inception'):   
        K.clear_session()
        if model_name=='Inception':
            base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')        
            out = base_model.output
            out = GlobalAveragePooling2D()(out)
            self.model = Model(inputs=base_model.input, outputs=out)
        elif model_name=='VGG':
            self.model = applications.VGG16(include_top=False, weights='imagenet')
        else:
            raise ValueError("Invalid model")
        self.session = K.get_session()
        
    def load_candidate_image(self, idx,img_folder='validation', noise=0, mode='gaussian', dataset='cats_dogs', model='VGG', array_of_pics=False):
        from skimage.util import random_noise
        from PIL import Image

        if dataset=='cats_dogs':
            filenames_train = np.load('filenames_train.npy')    
            filenames_validation = np.load('filenames_validation.npy')    
            img_width, img_height = 150, 150
            root_path = 'data/'
            size = [1,8192]
            model = 'VGG'
            print('Using VGG Model')
        elif dataset=='flowers' and model=='Squeeze':
            filenames_train = np.load('data/flower_photos/filenames_train-squeeze.npy')    
            filenames_validation = np.load('data/flower_photos/filenames_validation-squeeze.npy')    
            img_width, img_height = 227, 227
            root_path = 'data/flower_photos/'
            size = [1,13,13,512]
        elif dataset=='flowers' and model=='Inception':
            filenames_train = np.load('data/flower_photos/filenames_train-inception.npy')    
            filenames_validation = np.load('data/flower_photos/filenames_validation-inception.npy')    
            img_width, img_height = 227, 227
            root_path = 'data/flower_photos/'
            size = [1,2048]        
        else:
            raise ValueError("Invalid dataset or model")

        if (array_of_pics):
            padding = 10
            paths = [root_path+'train'+"/"+filenames_train[i] for i in idx]
            images = [Image.open(p).resize((img_width, img_height)) for p in paths]
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)+2*padding
            max_height = max(heights)
            new_im = Image.new('RGB', (total_width, max_height),"white")

            x_offset = 0
            for im in images:
                new_im.paste(im, (x_offset,0))
                x_offset += im.size[0]+padding
            return new_im

        else:
            if (img_folder=='train'):
                filename = filenames_train[idx]
            elif (img_folder=='validation'):
                filename = filenames_validation[idx]
            else:
                raise ValueError("Invalid argument for paraemter: folder")
            #print("Opening file:"+root_path+img_folder+"/"+filename)
            pic = Image.open(root_path+img_folder+"/"+filename)
            pic = pic.resize((img_width, img_height))
            pic = np.array(pic)/255
            #pic = np.array(pic)
            if (mode=='binary'):
                pic += np.random.randint(2, size=pic.shape)*noise
                pic = np.minimum(np.maximum(pic,0),1)
            else:
                pic = random_noise(pic, mode=mode, var=noise, clip=True)
            pic = pic.reshape(-1,img_width, img_height, 3)
            features = self.get_bottleneck_representation(pic,model_name=model)
            return pic.squeeze(), features.reshape(size)
                         
    def pop_layer(self, model):
        if not model.outputs:
            raise Exception('Sequential model cannot be popped: model is empty.')

        model.layers.pop()
        if not model.layers:
            model.outputs = []
            model.inbound_nodes = []
            model.outbound_nodes = []
        else:
            model.layers[-1].outbound_nodes = []
            model.outputs = [model.layers[-1].output]
        model.built = False


    def get_bottleneck_representation(self, pic, model_name='VGG'):
        from keras.preprocessing.image import ImageDataGenerator
        img_width, img_height = 299, 299
        datagen = ImageDataGenerator(rescale=1. / 255)
        
        K.set_session(self.session) 
        #features = self.model.predict(pic)
        for x_batch in datagen.flow(pic, None, batch_size=1):
            #print(x_batch.shape)
            features = self.model.predict(x_batch*255)
            #print(np.allclose(x_batch, pic))
            #print(np.allclose(x_batch, pic/255))
            break
        K.clear_session()
        return features
    
        #sess2 = tf.Session()
        #if (model_name=='VGG'):
        #    model = applications.VGG16(include_top=False, weights='imagenet')
        #if (model_name=='Inception'):
        #    base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')        
        #    out = base_model.output
        #    out = GlobalAveragePooling2D()(out)
        #    model = Model(inputs=base_model.input, outputs=out)
        #elif (model_name=='Squeeze'):
        #    model = SqueezeNet(include_top=False) #the include_top doesn't actually seem to do anything
        #    pop_layer(model)
        #    pop_layer(model)
        #    pop_layer(model)
        #    pop_layer(model)
        #else:
        #    raise ValueError("Invalid argument for paraemter: model_name")


    def pgti(self,grads, pic, model='Inception'):
        from keras import backend as K
        K.set_session(self.session)
        #K.set_learning_phase(1)

        pic = pic.reshape(1, 227, 227, 3)
        grads = grads.reshape(2048)
        
        with K.get_session().graph.as_default():

            gradient_wrt_features = tf.placeholder(tf.float32, shape=[2048])
            base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
            out = base_model.output
            out = GlobalAveragePooling2D()(out)
            model = Model(inputs=base_model.input, outputs=out)

            flattened_features = tf.reshape(model.output, [-1])
            product_of_gradient_and_features = tf.multiply(gradient_wrt_features, flattened_features)
            product_of_gradient_and_features = tf.reduce_sum(product_of_gradient_and_features)
            gradient_op = tf.gradients(product_of_gradient_and_features, model.input)

            K.set_learning_phase(0)
            gradient = self.session.run(gradient_op, feed_dict={model.input:pic, gradient_wrt_features:grads})

        K.clear_session()
        return gradient     
    
def propagate_gradients_to_input(grads, pic, model='Inception'):
    from keras import backend as K
    #set learning phase
    #tf.reset_default_graph()
    sess2 = tf.Session()
    K.set_session(sess2)
    #K.set_learning_phase(0)
    
    pic = pic.reshape(1, 227, 227, 3)
    grads = grads.reshape(2048)
    
    gradient_wrt_features = tf.placeholder(tf.float32, shape=[2048])
    base_model = applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    model = Model(inputs=base_model.input, outputs=out)

    flattened_features = tf.reshape(model.output, [-1])
    product_of_gradient_and_features = tf.multiply(gradient_wrt_features, flattened_features)
    product_of_gradient_and_features = tf.reduce_sum(product_of_gradient_and_features)
    gradient_op = tf.gradients(product_of_gradient_and_features, model.input)
    
    K.set_learning_phase(0)
    gradient = sess2.run(gradient_op, feed_dict={model.input:pic, gradient_wrt_features:grads})
    
    K.clear_session()
    del model
    return gradient 
    
def make_prediction(sess, X, y, features):
    prob = sess.run(y, feed_dict={X:features})
    if (prob<0.5):
        return prob, 0
    else:
        return prob, 1        
    

def get_nearest_neighbors(X_train, X_test, N):
    #X_test = X_test.flatten()
    dist_2 = np.sum((X_train - X_test)**2, axis=1)

    idxs_most = np.argpartition(dist_2, -N, axis=0)[-N:]
    idxs_most = idxs_most[np.argsort(dist_2[idxs_most])]

    non_influences = np.square(np.zeros(dist_2.shape)-dist_2)
    idxs_neutral = np.argpartition(non_influences, -N, axis=0)[-N:]
    idxs_neutral = idxs_neutral[np.argsort(non_influences[idxs_neutral])]

    idxs_least = np.argpartition(dist_2, N, axis=0)[:N]
    idxs_least = idxs_least[np.argsort(dist_2[idxs_least])]
    
    return (idxs_least, idxs_neutral, idxs_most)

def compute_confidence_from_probability(prob, label):
    prob = float(prob)
    prob = 1-np.absolute(label-prob)
    return prob

def show_influential_images(infl, train_shuffle_idx, valid_shuffle_idx, N = 5, top=True, middle=True, bottom=True, dataset='cats_dogs', model='VGG', array_of_pics=False, basis='influence', X_train=None, X_test=None):
    from PIL import Image
    
    if dataset=='cats_dogs':
        filenames_train = np.load('filenames_train.npy')    
        filenames_validation = np.load('filenames_validation.npy')    
        img_width, img_height = 150, 150
        root_path = 'data/'
        size = [1,8192]
    elif dataset=='flowers' and model=='Squeeze':
        filenames_train = np.load('data/flower_photos/filenames_train-squeeze.npy')    
        filenames_validation = np.load('data/flower_photos/filenames_validation-squeeze.npy')    
        img_width, img_height = 227, 227
        root_path = 'data/flower_photos/'
        size = [1,13,13,512]
    elif dataset=='flowers' and model=='Inception':
        filenames_train = np.load('data/flower_photos/filenames_train-inception.npy')    
        filenames_validation = np.load('data/flower_photos/filenames_validation-inception.npy')    
        img_width, img_height = 227, 227
        root_path = 'data/flower_photos/'
        size = [1,2048]        
    else:
        raise ValueError("Invalid dataset or model")
    
    if (basis=='influence'):
        most, neutral, least = infl.get_most_neutral_least(N=N)        
        influences = infl.influences
    elif (basis=='nearest-neighbor'):
        most, neutral, least = get_nearest_neighbors(X_train, X_test, N=N)        
    else:
        raise ValueError("Invalid similarity basis")
        
    if (array_of_pics):
        padding = 10
        paths = [root_path+'train'+"/"+filenames_train[train_shuffle_idx[i]] for i in most]
        images = [Image.open(p).resize((img_width, img_height)) for p in paths]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)+2*padding
        max_height = max(heights)
        new_im = Image.new('RGB', (total_width, max_height),"white")

        x_offset = 0
        for im in images:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]+padding
        return new_im
    
    def show_img_from_idx(idx, img_folder,i=None):
        if (img_folder=='train'):
            true_idx = train_shuffle_idx[idx] #confirm this is correct
            filename = filenames_train[true_idx]
        elif (img_folder=='validation'):
            true_idx = valid_shuffle_idx[idx] #confirm this is correct
            filename = filenames_validation[true_idx]
        else:
            raise ValueError("Invalid argument for parameter: folder")
            
        pic = Image.open(root_path+img_folder+"/"+filename)
        pix = np.array(pic)
        plt.imshow(pix)
        if not(i is None):
            plt.title("#" + str(i) +", influence: " + str(np.round(influences[idx],2)))            
        else:
            if (basis=='influence'):
                plt.title(str(true_idx)+" Influential: " + str(np.round(influences[idx],2)))
    
    if (top):
        plt.figure(figsize=[15,4])
        for i, idx in enumerate(most[::-1]):
            plt.subplot(1,N,i+1)
            show_img_from_idx(idx, img_folder='train',i=i+1)
    if (middle):
        plt.figure(figsize=[15,4])
        for i, idx in enumerate(neutral):
            plt.subplot(1,5,i+1)
            show_img_from_idx(idx, img_folder='train')
    if (bottom):
        plt.figure(figsize=[15,4])
        for i, idx in enumerate(least):
            plt.subplot(1,5,i+1)
            show_img_from_idx(idx, img_folder='train')
