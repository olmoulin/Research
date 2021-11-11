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

from deer.base_classes import Environment

def generate_results_comparison():
    OAMT_training_time = np.load('One_agent_multi_training_time.npy')
    MAOT_training_time = np.load('Multi_agent_one_training_time.npy')
    plt.title('Training time comparison')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[0, 50, 100, 150, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],rotation=90)
    plt.plot(MAOT_training_time, color='blue', label='eco-system duration')
    plt.plot(OAMT_training_time, color='red', label='one-agent duration')
    plt.legend()
    plt.savefig('Architecture_comparison_duration.png')
    plt.close()
    OAMT_forget = np.load('One_agent_multi_training_forget.npy')
    MAOT_forget = np.load('Multi_agent_one_training_forget.npy')
    plt.title('% accuracy on learned environments')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[0, 50, 100, 150, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],rotation=90)
    plt.plot(MAOT_forget, color='blue', label='eco-system accuracy')
    plt.plot(OAMT_forget, color='red', label='one-agent accuracy')
    plt.legend()
    plt.savefig('Architecture_comparison_forget.png')
    plt.close()
    OAMT_general = np.load('One_agent_multi_training_general.npy')
    MAOT_general = np.load('Multi_agent_one_training_general.npy')
    plt.title('% generalization on new environments')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[0, 50, 100, 150, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],rotation=90)
    plt.plot(MAOT_general, color='blue', label='eco-system accuracy')
    plt.plot(OAMT_general, color='red', label='one-agent accuracy')
    plt.legend()
    plt.savefig('Architecture_comparison_general.png')
    plt.close()
    OAMT_access = np.load('One_agent_multi_training_access.npy')
    MAOT_access = np.load('Multi_agent_one_training_access.npy')
    plt.title('access to environments')
    plt.xticks([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],[0, 50, 100, 150, 200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000],rotation=90)
    plt.plot(MAOT_access, color='blue', label='eco-system access')
    plt.plot(OAMT_access, color='red', label='one-agent access')
    plt.legend()
    plt.savefig('Architecture_comparison_access.png')
    plt.close()

def main():
    generate_results_comparison()

if __name__ == "__main__":
    main()
