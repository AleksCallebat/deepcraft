from __future__ import print_function
from __future__ import print_function
from __future__ import absolute_import, division, print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
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
# Targetted reward
REWARD_TARGET = 30 if isFast else 200
# Averaged over these these many episodes
BATCH_SIZE_BASELINE = 20 if isFast else 50
MEMORY_CAPACITY=100000
H = 64 # hidden layer size
STATE_COUNT=2
ACTION_COUNT=4
MIN_EPSILON=0.01
LAMBDA = 0.01 # speed of decay
MAX_EPSILON=1
GAMMA = 0.001 # discount factor
BATCH_SIZE=BATCH_SIZE_BASELINE
from builtins import range
import sys
from utils import *

def run(argv=['']):
    if "MALMO_XSD_PATH" not in os.environ:
        print("Please set the MALMO_XSD_PATH environment variable.")
        return

    malmoutils.fix_print()

    agent_host = MalmoPython.AgentHost()
    malmoutils.parse_command_line(agent_host, argv)

    my_mission = MalmoPython.MissionSpec()
    my_mission.timeLimitInSeconds( 10 )
    my_mission.requestVideo( 320, 240 )
    my_mission.rewardForReachingPosition( 19.5, 0.0, 19.5, 100.0, 1.1 )

    my_mission_record = malmoutils.get_default_recording_object(agent_host, "saved_data")

    # client_info = MalmoPython.ClientInfo('localhost', 10000)
    client_info = MalmoPython.ClientInfo('127.0.0.1', 10000)
    pool = MalmoPython.ClientPool()
    pool.add(client_info)

    experiment_id = str(uuid.uuid1())
    print("experiment id " + experiment_id)

    max_retries = 3
    max_response_time = 60  # seconds

    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, pool, my_mission_record, 0, experiment_id)
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)

    print("Waiting for the mission to start", end=' ')
    start_time = time.time()
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        print(".", end="")
        time.sleep(0.1)
        if time.time() - start_time > max_response_time:
            print("Max delay exceeded for mission to begin")
            restart_minecraft(world_state, agent_host, client_info, "begin mission")
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    print()

    last_delta = time.time()
    # main loop:
    while world_state.is_mission_running:
        agent_host.sendCommand( "move 1" )
        agent_host.sendCommand( "turn " + str(random.random()*2-1) )
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        print("video,observations,rewards received:",world_state.number_of_video_frames_since_last_state,world_state.number_of_observations_since_last_state,world_state.number_of_rewards_since_last_state)
        if (world_state.number_of_video_frames_since_last_state > 0 or
           world_state.number_of_observations_since_last_state > 0 or
           world_state.number_of_rewards_since_last_state > 0):
            last_delta = time.time()
        else:
            if time.time() - last_delta > max_response_time:
                print("Max delay exceeded for world state change")
                restart_minecraft(world_state, agent_host, client_info, "world state change")
        for reward in world_state.rewards:
            print("Summed reward:",reward.getValue())
        for error in world_state.errors:
            print("Error:",error.text)
        for frame in world_state.video_frames:
            print("Frame:",frame.width,'x',frame.height,':',frame.channels,'channels')
            image = Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) ) # to convert to a PIL image
            import numpy as np
            print(np.array(image))
    print("Mission has stopped.")

if __name__ == "__main__":
    run(sys.argv)
