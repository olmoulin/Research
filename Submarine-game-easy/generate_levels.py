import numpy as np
np.random.seed(1231231)

import tensorflow as tf
tf.random.set_seed(1231231)

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='cpu')

from tensorflow.keras.models import Sequential, save_model, load_model, model_from_json
from tensorflow.keras.layers import Dense,PReLU,Input,Conv2D,Flatten
from tensorflow.keras.optimizers import Adam, RMSprop

from datetime import datetime
from scipy.special import softmax
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image
from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.q_net_keras import MyQNetwork
import deer.experiment.base_controllers as bc
from deer.policies import EpsilonGreedyPolicy

import moviepy.editor as mpy
import time

def main():
    for seed in range(0,10000):
        np.random.seed(seed)
        Marine=np.zeros((11,25))
        Marine[0,0:20]= 1
        Marine[10,0:20]=1
        for i in range(0,25):
            if i%2==0 and i>0:
                for j in range(0,3):
                    rnd = np.random.randint(0,11)
                    Marine[rnd,i]=1
            if i>=20:
                Marine[0:11,i]=1
        np.save('level/level-'+str(seed)+'.npy',Marine)



if __name__ == "__main__":
    main()
