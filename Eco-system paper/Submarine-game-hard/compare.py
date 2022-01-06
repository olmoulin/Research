import numpy as np
np.random.seed(1231231)

from datetime import datetime
from scipy.special import softmax
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os, sys, time, datetime, json, random
import gym
from PIL import Image

import time

def generate_results_comparison():
    plt.title('Training time comparison')
    plt.xticks([0,4,8,12,16,20],[0, 200,400,600,800,1000],rotation=0)
    plt.ylabel('seconds')
    plt.xlabel('# of environment trained')
    for i in range(5):
        OAMT_training_time = np.load('One_agent_multi_training_time'+str(i)+'.npy')
        MAOT_training_time = np.load('Multi_agent_one_training_time'+str(i)+'.npy')
        plt.plot(MAOT_training_time, color='blue', label='eco-system duration')
        plt.plot(OAMT_training_time, color='red', label='one-agent duration')
    red_patch = mpatches.Patch(color='red', label='single-agent')
    blue_patch = mpatches.Patch(color='blue', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_duration_easy.png')
    plt.close()

    plt.title('% accuracy on learned environments')
    plt.xticks([0,4,8,12,16,20],[0, 200,400,600,800,1000],rotation=0)
    plt.ylabel('%')
    plt.xlabel('# of environment trained')
    for i in range(5):    
        OAMT_forget = np.load('One_agent_multi_training_forget'+str(i)+'.npy')
        MAOT_forget = np.load('Multi_agent_one_training_forget'+str(i)+'.npy')
        plt.plot(MAOT_forget*100, color='blue', label='eco-system accuracy')
        plt.plot(OAMT_forget*100, color='red', label='one-agent accuracy')
    red_patch = mpatches.Patch(color='red', label='single-agent')
    blue_patch = mpatches.Patch(color='blue', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_forget_easy.png')
    plt.close()
    
    plt.title('% generalization on new environments')
    plt.xticks([0,4,8,12,16,20],[0, 200,400,600,800,1000],rotation=0)
    plt.ylabel('%')
    plt.xlabel('# of environment trained')    
    for i in range(5):
        OAMT_general = np.load('One_agent_multi_training_general'+str(i)+'.npy')
        MAOT_general = np.load('Multi_agent_one_training_general'+str(i)+'.npy')
        plt.plot(MAOT_general, color='blue', label='eco-system accuracy')
        plt.plot(OAMT_general, color='red', label='single-agent accuracy')
    red_patch = mpatches.Patch(color='red', label='single-agent')
    blue_patch = mpatches.Patch(color='blue', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_general_easy.png')
    plt.close()
    
    plt.title('access to environments')    
    plt.xticks([0,4,8,12,16,20],[0, 200,400,600,800,1000],rotation=0)    
    plt.ylabel('# accesses')
    plt.xlabel('# of environment trained')
    for i in range(5):
        OAMT_access = np.load('One_agent_multi_training_access'+str(i)+'.npy')
        MAOT_access = np.load('Multi_agent_one_training_access'+str(i)+'.npy')
        plt.plot(MAOT_access, color='blue', label='eco-system access')
        plt.plot(OAMT_access, color='red', label='single-agent access')
    red_patch = mpatches.Patch(color='red', label='single-agent')
    blue_patch = mpatches.Patch(color='blue', label='eco-system')
    plt.legend(handles=[red_patch,blue_patch])
    plt.savefig('Architecture_comparison_access_easy.png')
    plt.close()

def main():
    generate_results_comparison()

if __name__ == "__main__":
    main()
