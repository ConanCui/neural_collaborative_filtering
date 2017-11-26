'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
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
from keras.regularizers import l1, l2
from keras.models import Sequential, Model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Merge, Flatten, Dropout
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from evaluate import evaluate_model
from Dataset import Dataset
from Dataset import get_train_instances, init_logging
from time import time
import logging
import sys
import GMF, MLP
import argparse



#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
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
    parser.add_argument('--num_factors', type=int, default=20,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[160,80,40,20]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--reg_mf', type=float, default=0,
                        help='Regularization for MF embeddings.')                    
    parser.add_argument('--reg_layers', nargs='?', default='[0,0,0,0]',
                        help="Regularization for each MLP layer. reg_layers[0] is the regularization for embeddings.")
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    return parser.parse_args()

def get_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer='uniform', embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer='uniform', embeddings_regularizer = l2(reg_mf), input_length=1)

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]/2, name = "mlp_embedding_user",
                                   embeddings_initializer='uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]/2, name = 'mlp_embedding_item',
                                   embeddings_initializer='uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
    mf_vector = keras.layers.Multiply()([mf_user_latent, mf_item_latent]) # element-wise multiply

    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = keras.layers.Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in xrange(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]), activation='relu', name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = keras.layers.Concatenate()([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, kernel_initializer='lecun_uniform', name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)
    
    return model

def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)
    
    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)
    
    # MLP layers
    for i in xrange(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' %i).get_weights()
        model.get_layer('layer%d' %i).set_weights(mlp_layer_weights)
        
    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5*new_weights, 0.5*new_b])    
    return model

if __name__ == '__main__':
    args = parse_args()
    num_epochs = args.epochs
    batch_size = args.batch_size
    mf_dim = args.num_factors
    layers = eval(args.layers)
    reg_mf = args.reg_mf
    reg_layers = eval(args.reg_layers)
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain
    init_logging("log/NeuMF.log")

    evaluation_threads = 1#mp.cpu_count()
    logging.info("NeuMF arguments: %s " %(args))
    model_out_file = 'Pretrain/%s_NeuMF_%d_%s_%d.h5' %(args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.ratio)
    train, test = dataset.trainMatrix, dataset.testMatrix
    num_users, num_items = train.shape
    user_traininput, item_traininput, trainlabels = get_train_instances(train)
    user_testinput, item_testinput, testlabels = get_train_instances(test)
    logging.info("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          %(time()-t1, num_users, num_items, train.nnz, test.nnz))
    
    # Build model
    model = get_model(num_users, num_items, mf_dim, layers, reg_layers, reg_mf)
    if learner.lower() == "adagrad": 
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='mean_squared_error')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='mean_squared_error')
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='mean_squared_error')
    
    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMF.get_model(num_users,num_items,mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users,num_items, layers, reg_layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        logging.info("Load pretrained GMF (%s) and MLP (%s) models done. " %(mf_pretrain, mlp_pretrain))


    # Training model
    best_testRmse, best_iter = 10e3, -1
    for epoch in xrange(num_epochs):
        t1 = time()
        # Generate training instances
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
