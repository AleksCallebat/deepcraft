from __future__ import print_function
from __future__ import print_function
from __future__ import absolute_import, division, print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
import csv
import logging
import os
import random
from PIL import Image
import numpy as np
import sys,math
import cntk as C
import time
import tkinter as tk

isFast=True
# Averaged over these these many episodes
BATCH_SIZE_BASELINE = 64
MEMORY_CAPACITY=100000000
H = 64 # hidden layer size
STATE_COUNT=2306
ACTION_COUNT=4
#MIN_EPSILON=0.01
#LAMBDA = 0.01 # speed of decay
MAX_EPSILON=1
GAMMA = 0.33 # discount factor
BATCH_SIZE=BATCH_SIZE_BASELINE
from builtins import range
import sys
import os
import random
import time
import uuid
from PIL import Image

# Allow MalmoPython to be imported both from an installed
# malmo module and (as an override) separately as a native library.
try:
    import MalmoPython
    import malmoutils
except ImportError:
    import malmo.MalmoPython as MalmoPython
    import malmo.malmoutils as malmoutils


class MissionTimeoutException(Exception):
    pass


def restart_minecraft(world_state, agent_host, client_info, message):
    """"Attempt to quit mission if running and kill the client"""
    if world_state.is_mission_running:
        agent_host.sendCommand("quit")
        time.sleep(10)
    agent_host.killClient(client_info)
    raise MissionTimeoutException(message)


class Brain:
    def __init__(self):
        self.params = {}
        self.model, self.trainer, self.loss = self._create()
        try:
            self.model=C.load_model("model.model")
        except:
            print("couldn't load model")
    def _create(self):
        observation = C.sequence.input_variable(STATE_COUNT, np.float32, name="s")
        q_target = C.sequence.input_variable(ACTION_COUNT, np.float32, name="q")

        # Following a style similar to Keras
        l1 = C.layers.Dense(64, activation=C.relu)
        #l2 = C.layers.Convolution(16,reduction_rank=0 ,activation=C.relu)
        l3 = C.layers.Dense(ACTION_COUNT)
        unbound_model = C.layers.Sequential([l1,  l3])
        model = unbound_model(observation)

        self.params = dict(W1=l1.W, b1=l1.b, W3=l3.W,b3=l3.b)

        # loss='mse'
        loss = C.reduce_mean(C.square(model - q_target), axis=0)
        meas = C.reduce_mean(C.square(model - q_target), axis=0)

        # optimizer
        lr = 0.00025
        lr_schedule = C.learning_parameter_schedule(lr)
        learner = C.sgd(model.parameters, lr_schedule, gradient_clipping_threshold_per_sample=10)
        trainer = C.Trainer(model, (loss, meas), learner)

        # CNTK: return trainer and loss as well
        return model, trainer, loss

    def train(self, x, y, epoch=1, verbose=0):
        #self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)
        arguments = dict(zip(self.loss.arguments, [x,y]))
        updated, results =self.trainer.train_minibatch(arguments, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval([s])

class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity
        samples=[]

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def length(self):
        return(len(self.samples))


#if __name__ == "__main__":
#    run(sys.argv)
