#!/usr/bin/env python
from __future__ import print_function

import argparse
import skimage as skimage
from skimage import transform, color, exposure
from skimage.transform import rotate
from skimage.viewer import ImageViewer
import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.append("games/flappy/")
sys.path.append("games/Fake-Arkanoid/")
#import wrapped_flappy_bird as game_flappy
import wrapped_arkanoid as game_arkanoid
import random
import numpy as np
import numpy.ma as ma
from collections import deque
import matplotlib.cm as cm
import pylab as pl

import json
from keras import initializers
from keras import backend as K
from keras.initializers import normal, identity
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD , Adam
import tensorflow as tf

from mpl_toolkits.axes_grid1 import make_axes_locatable

GAME = 'bird' # the name of the game being played for log files
CONFIG = 'nothreshold'
ACTIONS = 3 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVATION = 100. # timesteps to observe before training
EXPLORE = 5000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.01 # final value of epsilon
INITIAL_EPSILON = 0.3 # starting value of epsilon
EPSILON_DECAY = (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE
REPLAY_MEMORY = 2000 # number of previous transitions to remember
BATCH = 512 # size of minibatch
FRAME_PER_ACTION = 5
LEARNING_RATE = 0.0001
IMG_SHOW_STEP = 100

img_rows , img_cols = 80, 80
#Convert image into Black and white
img_channels = 4 #We stack 4 frames

class MyModel:
    def __init__(self):
        print("Now we build the model")
        #self.create_default_model()
        #self.create_simple_model()
        self.create_hard_model()
        print("We finish building the model")

    def create_hard_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same", input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.convout1 = Activation('relu')
        self.model.add(self.convout1)
        
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(32, (2, 2), strides=(1, 1), padding="same"))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dense(ACTIONS))
       
        adam = Adam(lr=LEARNING_RATE)
        self.model.compile(loss='mse',optimizer=adam)

    def create_default_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(32, (4, 4), strides=(2, 2), padding="same", input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.convout1 = Activation('relu')
        self.model.add(self.convout1)
        
        self.model.add(Conv2D(64, (4, 4), strides=(2, 2), padding="same"))
        self.model.add(Activation('relu'))
        
        self.model.add(Conv2D(64, (2, 2), strides=(1, 1), padding="same"))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dense(128))
        self.model.add(Activation('relu'))
        self.model.add(Dense(ACTIONS))
       
        adam = Adam(lr=LEARNING_RATE)
        self.model.compile(loss='mse',optimizer=adam)

    def create_simple_model(self):
        self.model = Sequential()

        self.model.add(Conv2D(16, (4, 4), strides=(2, 2), padding="same", input_shape=(img_rows,img_cols,img_channels)))  #80*80*4
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.convout1 = Activation('relu')
        self.model.add(self.convout1)
        
        self.model.add(Conv2D(16, (2, 2), strides=(2, 2), padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Activation('relu'))

        self.model.add(Flatten())
        self.model.add(Dense(64))
        self.model.add(Activation('relu'))
        self.model.add(Dense(ACTIONS))
       
        adam = Adam(lr=LEARNING_RATE)
        self.model.compile(loss='mse',optimizer=adam)

    def make_mosaic(self, imgs, nrows, ncols, border=1):
        """
        Given a set of images with all the same shape, makes a
        mosaic with nrows and ncols
        """
        nimgs = imgs.shape[0]
        imshape = imgs.shape[1:]
        print(imshape)
        
        mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                                ncols * imshape[1] + (ncols - 1) * border),
                                dtype=np.float32)
        
        paddedh = imshape[0] + border
        paddedw = imshape[1] + border
        for i in range(nimgs):
            row = int(np.floor(i / ncols))
            col = i % ncols
            
            mosaic[row * paddedh:row * paddedh + imshape[0],
                   col * paddedw:col * paddedw + imshape[1]] = imgs[i]
        return mosaic

    def nice_imshow(self, ax, data, vmin=None, vmax=None, cmap=None):
        """Wrapper around pl.imshow"""
        if cmap is None:
            cmap = cm.jet
        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
        pl.colorbar(im, cax=cax)
        pl.show()

    def show_conv(self, X):
        convout1_f = K.function(self.model.inputs, [self.convout1.output])
        C1 = convout1_f([X])
        C1 = np.squeeze(C1)

        pl.figure(figsize=(15, 15))
        pl.suptitle('convout1')
        self.nice_imshow(pl.gca(), self.make_mosaic(C1, 2, 10), cmap=cm.binary)

    def trainNetwork(self, args):
        # open up a game state to communicate with emulator
        #game_state = game_flappy.GameState()
        game_state = game_arkanoid.GameState(True)# if args['mode'] == 'Run' else False)

        # store the previous observations in replay memory
        D = deque()

        # get the first state by doing nothing and preprocess the image to 80x80x4
        do_nothing = np.zeros(ACTIONS)
        do_nothing[0] = 1
        x_t, r_0, terminal = game_state.frame_step(do_nothing)

        x_t = skimage.color.rgb2gray(x_t)
        x_t = skimage.transform.resize(x_t,(img_rows,img_cols), mode='constant')
        #x_t = skimage.exposure.rescale_intensity(x_t,out_range=(0,255))

        s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
        #print (s_t.shape)

        #In Keras, need to reshape
        s_t = s_t.reshape(1, s_t.shape[0], s_t.shape[1], s_t.shape[2])  #1*80*80*4

        if os.path.isfile("model.h5"):
            print ("Now we load weight")
            self.model.load_weights("model.h5")
            adam = Adam(lr=LEARNING_RATE)
            self.model.compile(loss='mse',optimizer=adam)
            print ("Weight load successfully")

        if args['mode'] == 'Run':
            OBSERVE = 999999999    #We keep observe, never train
            epsilon = 0   
        else:                       #We go to training mode
            OBSERVE = OBSERVATION
            epsilon = INITIAL_EPSILON
            if os.path.isfile("model.h5"):
                epsilon = FINAL_EPSILON

        t = 0
        loss = 0
        prev_points = 0
        while (True):
            Q_sa = self.model.predict(s_t)       #input a stack of 4 images, get the prediction
            action_index = 0
            r_t = 0
            a_t = np.zeros([ACTIONS])
            #choose an action epsilon greedy
            if random.random() <= epsilon:
                #print("----------Random Action----------")
                action_index = random.randrange(ACTIONS)
                a_t[action_index] = 1
            else:
                max_Q = np.argmax(Q_sa)
                action_index = max_Q
                a_t[max_Q] = 1

            #We reduced the epsilon gradually
            if epsilon > FINAL_EPSILON + EPSILON_DECAY and t > OBSERVE:
                epsilon -= EPSILON_DECAY

            #run the selected action and observed next state and reward
            terminal = False
            for _ in range(FRAME_PER_ACTION-1):
                _, _, terminal_aux = game_state.frame_step(a_t)
                terminal = terminal_aux or terminal

            x_t1_colored, points, terminal_aux = game_state.frame_step(a_t)
            terminal = terminal_aux or terminal

            r_t = points - prev_points if points >= prev_points else -50
            prev_points = points

            x_t1 = skimage.color.rgb2gray(x_t1_colored)
            x_t1 = skimage.transform.resize(x_t1,(img_rows,img_cols), mode='constant')
            #x_t1 = skimage.exposure.rescale_intensity(x_t1, out_range=(0, 255))

            x_t1 = x_t1.reshape(1, x_t1.shape[0], x_t1.shape[1], 1) #1x80x80x1
            s_t1 = np.append(x_t1, s_t[:, :, :, :3], axis=3)

            #if t == IMG_SHOW_STEP: self.show_conv(s_t1)
            # store the transition in D
            D.append((s_t, action_index, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            #only train if done observing
            if t>0 and t % OBSERVE == 0:
                #sample a minibatch to train on
                minibatch = random.sample(D, min(BATCH, len(D)))

                inputs = np.zeros((BATCH, s_t.shape[1], s_t.shape[2], s_t.shape[3]))   #32, 80, 80, 4
                #print (inputs.shape)
                targets = np.zeros((inputs.shape[0], ACTIONS))                         #32, 2

                #Now we do the experience replay
                for i in range(0, len(minibatch)):
                    state_t = minibatch[i][0]
                    action_t = minibatch[i][1]   #This is action index
                    reward_t = minibatch[i][2]
                    state_t1 = minibatch[i][3]
                    terminal = minibatch[i][4]
                    # if terminated, only equals reward

                    inputs[i:i + 1] = state_t    #I saved down s_t

                    targets[i] = self.model.predict(state_t)  # Hitting each buttom probability
                    Q_sa = self.model.predict(state_t1)

                    if terminal:
                        targets[i, action_t] = reward_t
                    else:
                        targets[i, action_t] = reward_t + GAMMA * np.max(Q_sa)
                    
                # targets2 = normalize(targets)
                loss = self.model.train_on_batch(inputs, targets)

            s_t = s_t1
            t = t + 1

            # save progress every 1000 iterations
            if t % 1000 == 0:
                print("Now we save model")
                self.model.save_weights("model.h5", overwrite=True)
                with open("model.json", "w") as outfile:
                    json.dump(self.model.to_json(), outfile)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            print("TIMESTEP", t, "/ STATE", state, \
                    "/ EPSILON %.5f "% epsilon, "/ ACTION", action_index, "/ R", r_t, \
                    "\t/ Q_MAX %.5f "% np.max(Q_sa), "/ Loss %.5f"% loss)
            

        print("Episode finished!")
        print("************************")

def playGame(args):
    model = MyModel()
    model.trainNetwork(args)

def main():
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-m','--mode', help='Train / Run', required=True)
    args = vars(parser.parse_args())
    playGame(args)

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)
    main()
