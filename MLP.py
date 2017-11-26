'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import keras
from keras import backend as K
from keras import initializers
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.constraints import maxnorm
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from Dataset import get_train_instances, init_logging
from time import time
import logging
import sys
import argparse
import multiprocessing as mp


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--ratio', type=float, default='0.4',
                        help='the percent of data for training.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[160,80,40,20]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each layer")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=0,
                        help='Whether to save the trained model.')
    return parser.parse_args()

def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = 'user_embedding',
                                   embeddings_initializer='uniform', embeddings_regularizer = l2(reg_layers[0]),                                        input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'item_embedding',
                                   embeddings_initializer='uniform', embeddings_regularizer = l2(reg_layers[0]),                                         input_length=1)
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = keras.layers.Concatenate()([user_latent,item_latent])
    # MLP layers
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, kernel_initializer='lecun_uniform', name = 'prediction')(vector)
    
    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)
    
    return model

if __name__ == '__main__':
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    reg_layers = eval(args.reg_layers)
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    init_logging("log/MLP.log")
    
    topK = 10
    evaluation_threads = 1 #mp.cpu_count()
    logging.info("MLP arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' %(args.dataset, args.layers, time())
    
    # Loading data
    t1 = time()
    dataset = Dataset(args.ratio)
    train, validation, test = dataset.trainMatrix, dataset.validationMatrix, dataset.testMatrix
    num_users, num_items = train.shape
    user_traininput, item_traininput, trainlabels = get_train_instances(train)
    user_validationinput, item_validationinput, validationlabels = get_train_instances(validation)
    user_testinput, item_testinput, testlabels = get_train_instances(test)
    logging.info("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #validation= %d, #test=%d"
          %(time()-t1, num_users, num_items, train.nnz, validation.nnz, test.nnz))
    
    # Build model
    model = get_model(num_users, num_items, layers, reg_layers)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='mean_squared_error')


        # Train model
    best_testRmse, best_iter = 10e3, -1
    for epoch in xrange(epochs):
        t1 = time()

        # Training
        hist = model.fit([np.array(user_traininput), np.array(item_traininput)],  # input
                         np.array(trainlabels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()

        # Evaluation
        if epoch % verbose == 0:

            testloss = model.evaluate([np.array(user_testinput), np.array(item_testinput)], np.array(testlabels),
                                      verbose=0)
            trainloss = hist.history['loss'][0]
            testRmse = np.sqrt(testloss)
            logging.info('Iteration %d [%.1f s]: train_loss = %.4f, test_loss = %.4f, test_Rmse = %.4f, [%.1f s]'
                  % (epoch, t2 - t1, trainloss, testloss, testRmse, time() - t2))
            if testRmse < best_testRmse:
                best_testRmse, best_iter = testRmse, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    logging.info("End. Best Iteration %d:  test_Rmse = %.4f. " %(best_iter, best_testRmse))
    if args.out > 0:
        logging.info("The best MLP model is saved to %s" %(model_out_file))
