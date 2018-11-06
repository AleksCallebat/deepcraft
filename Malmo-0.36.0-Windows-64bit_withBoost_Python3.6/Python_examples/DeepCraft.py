from __future__ import print_function
from __future__ import absolute_import, division, print_function
from future import standard_library
standard_library.install_aliases()
from builtins import range
from builtins import object
import MalmoPython
import json
import logging
from utils import *
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
#REWARD_TARGET = 30 if isFast else 200
# Averaged over these these many episodes
MEMORY_CAPACITY=100000000
H = 64 # hidden layer size
STATE_COUNT=2306
ACTION_COUNT=4
MIN_EPSILON=0.05
LAMBDA = 0.3 # speed of decay
MAX_EPSILON=0.2
GAMMA = 0.25 # discount factor
BATCH_SIZE=BATCH_SIZE_BASELINE
sample_ratio=10

class TabQAgent(object):
    """Tabular Q-learning agent for discrete state/action spaces."""
    def __init__(self):
        self.brain=Brain()
        self.memory=Memory(MEMORY_CAPACITY)
        self.epsilon = 1 # chance of taking a random action instead of the best
        self.steps=0
        self.logger = logging.getLogger(__name__)
        if False: # True if you want to see more information
            self.logger.setLevel(logging.DEBUG)
        else:
            self.logger.setLevel(logging.INFO)
        self.logger.handlers = []
        self.logger.addHandler(logging.StreamHandler(sys.stdout))

        self.actions = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
        self.q_table = {}
        self.canvas = None
        self.root = None

    def updateQTable( self, reward, current_state ):
        """Change q_table to reflect what we have learnt."""
        state_to_draw="%d:%d" % (int(self.prev_s[0]), int(self.prev_s[1]))
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        old_q = self.q_table[state_to_draw][self.prev_a]
        if self.memory.length()>100:
            print(self.brain.predict(self.prev_s))
            pred=self.brain.predict(self.prev_s)[0][0]
            new_q = pred[int(self.prev_a)]
        else:
            new_q=old_q        
        # assign the new action value to the Q-table
        self.q_table[state_to_draw][self.prev_a] = new_q
        
    def updateQTableFromTerminatingState( self, reward ):
        """Change q_table to reflect what we have learnt, after reaching a terminal state."""
        
        # retrieve the old action value from the Q-table (indexed by the previous state and the previous action)
        state_to_draw="%d:%d" % (int(self.prev_s[0]), int(self.prev_s[1]))
        old_q = self.q_table[state_to_draw][self.prev_a]
        if self.memory.length()>20 :
            pred=self.brain.predict(self.prev_s)[0][0]
            new_q = pred[int(self.prev_a)]
        else:
            new_q=old_q
        # assign the new action value to the Q-table
        self.q_table[state_to_draw][self.prev_a] = new_q
        
    def act(self, current_s, agent_host, current_r ):
        """take 1 action in response to the current world state"""
        pos = "%d:%d" % (int(current_s[0]), int(current_s[1]))
        
        #self.logger.debug("State: %s (x = %.2f, z = %.2f)" % (current_s, float(obs[u'XPos']), float(obs[u'ZPos'])))
        if pos not in self.q_table:
            self.q_table[pos] = ([0] * len(self.actions))

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTable( current_r, pos )

        self.drawQ( curr_x =int(current_s[0]), curr_y = int(current_s[1]) )

        # select the next action
        rnd = random.random()
        if rnd < self.epsilon or self.memory.length()<20:
            a = random.randint(0, len(self.actions) - 1)
            self.logger.info("Random action: %s" % self.actions[a])
        else:
            #print("current s",current_s)
            temp=self.brain.predict(current_s)[0]
            a,b=np.argmax(temp),np.max(temp)
            
            self.logger.info("Taking q action: %s" % self.actions[a])
            self.logger.info("this is the ponderation: %s" % temp)

        # try to send the selected action, only update prev_s if this succeeds
        try:
            agent_host.sendCommand(self.actions[a])
            self.prev_s = current_s
            self.prev_a = a

        except RuntimeError as e:
            self.logger.error("Failed to send command: %s" % e)

        return current_r

    def run(self, agent_host):
        """run the agent on the world"""

        total_reward = 0
        
        self.prev_s = None
        self.prev_a = None
        
        is_first_action = True
        
        # main loop:
        world_state = agent_host.getWorldState()
        while world_state.is_mission_running:
            current_r = 0
            
            if is_first_action:

                # wait until have received a valid observation
                while True:
                    time.sleep(0.2)
                    world_state = agent_host.getWorldState()


                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                    if world_state.is_mission_running and world_state.number_of_video_frames_since_last_state>0 and  len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        #Getting the position
                        obs_text = world_state.observations[-1].text
                        obs = json.loads(obs_text) # most recent observation
                        self.logger.debug(obs)
                        current_s=[int(obs[u'XPos']),int(obs[u'ZPos'])]
                        #Getting the frame
                        count=0
                        for frame in world_state.video_frames:
                            count+=1
                            image = np.array(Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) ))
                        for k in range(len(image)):
                            if k%sample_ratio==0:
                                for l in range(len(image[k])):
                                    if l%sample_ratio==0:
                                        for a in image[k][l]:
                                            current_s.append(a)
                        STATE_COUNT=len(current_s)
                      
                        self.prev_s=current_s
                        if len(self.prev_s)==2:
                            print("did not add the image as expected")
                        #print("obs",obs,"current_s)
                        total_reward += self.act(current_s, agent_host, current_r)
                        break
                    if not world_state.is_mission_running:
                        break
                is_first_action = False
            else:
                # wait for non-zero reward
                while world_state.is_mission_running and current_r == 0:
                    time.sleep(0.2)
                    world_state = agent_host.getWorldState()
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:
                        current_r += reward.getValue()
                # allow time to stabilise after action
                while True:
                    time.sleep(0.2)
                    world_state = agent_host.getWorldState()                    
                    if len(world_state.rewards)>1:
                        print("rewards",world_state.rewards)
                    for error in world_state.errors:
                        self.logger.error("Error: %s" % error.text)
                    for reward in world_state.rewards:                        
                        current_r += reward.getValue()
                    if world_state.is_mission_running and len(world_state.video_frames)>0 and len(world_state.observations)>0 and not world_state.observations[-1].text=="{}":
                        #getting pos
                        obs_text = world_state.observations[-1].text                        
                        obs = json.loads(obs_text)
                        states = [int(obs[u'XPos']), int(obs[u'ZPos'])]
                        #getting frames
                        frames = world_state.video_frames
                        for frame in frames :
                            image = np.array(Image.frombytes('RGB', (frame.width, frame.height), bytes(frame.pixels) ))
                        for k in range(len(image)):
                            if k%sample_ratio==0:
                                for l in range(len(image[k])):
                                    if l%sample_ratio==0:
                                        for a in image[k][l]:
                                            states.append(a)
                        if len(states)==2:
                            print("did not add the image as expected")
                        total_reward += self.act(states, agent_host, current_r)
                        if len(self.prev_s)==2:
                            print("issue in image collection")
                            break
                        self.observe([self.prev_s,self.prev_a,current_r,states])
                        self.prev_s=states
                        break
                    if not world_state.is_mission_running:
                        break
       
                    rnd = random.random()
                    if self.memory.length()>20 and rnd<0.3:
                        self.replay(STATE_COUNT=2306)


        # process final reward
        self.logger.debug("Final reward: %d" % current_r)
        total_reward += current_r

        # update Q values
        if self.prev_s is not None and self.prev_a is not None:
            self.updateQTableFromTerminatingState( current_r )            
            if len(self.prev_s)==2:
                print("ERRROR : checking inside observation",self.prev_s)
            else:
                self.observe([self.prev_s,self.prev_a,current_r,None])
        if self.memory.length()>20:
            self.replay(STATE_COUNT=2306)

        self.drawQ()
    
        return total_reward

     
    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
    
    def replay(self,STATE_COUNT):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)
        if batchLen==0:
            print("WARNING : replaying empty memory sample")

        no_state = np.zeros(STATE_COUNT)
        # CNTK: explicitly setting to float32
        for o in batch:
            if len(o)!=4:
                print("memory length issue",len(o))
                batch.remove(o)
                batchLen-=1
        states = np.array([ o[0] for o in batch ], dtype=np.float32)
        states_ = np.array([(no_state if o[3] is None else o[3]) for o in batch ], dtype=np.float32)
        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)
        x = np.zeros((batchLen, STATE_COUNT)).astype(np.float32)
        y = np.zeros((batchLen, ACTION_COUNT)).astype(np.float32)

        for i in range(batchLen):
            s, a, r, s_ = batch[i]

            # CNTK: [0] because of sequence dimension
            t = p[0][i]
            if s_ is None:
                t[a] = r
            else:
                #t[a] = r 
                t[a] = r + GAMMA * np.amax(p_[0][i])

            x[i] = s
            y[i] = t
        print("learning from mistakes")
        self.brain.train(x, y)

    def drawQ( self, curr_x=None, curr_y=None ):
        scale = 40
        world_x = 6
        world_y = 14
        if self.canvas is None or self.root is None:
            self.root = tk.Tk()
            self.root.wm_title("Q-table")
            self.canvas = tk.Canvas(self.root, width=world_x*scale, height=world_y*scale, borderwidth=0, highlightthickness=0, bg="black")
            self.canvas.grid()
            self.root.update()
        self.canvas.delete("all")
        action_inset = 0.1
        action_radius = 0.1
        curr_radius = 0.2
        action_positions = [ ( 0.5, action_inset ), ( 0.5, 1-action_inset ), ( action_inset, 0.5 ), ( 1-action_inset, 0.5 ) ]
        # (NSWE to match action order)
        min_value = -20
        max_value = 20
        for x in range(world_x):
            for y in range(world_y):
                s = "%d:%d" % (x,y)
                self.canvas.create_rectangle( x*scale, y*scale, (x+1)*scale, (y+1)*scale, outline="#fff", fill="#000")
                for action in range(4):
                    if not s in self.q_table:
                        continue
                    value = self.q_table[s][action]
                    color = int( 255 * ( value - min_value ) / ( max_value - min_value )) # map value to 0-255
                    color = max( min( color, 255 ), 0 ) # ensure within [0,255]
                    color_string = '#%02x%02x%02x' % (255-color, color, 0)
                    self.canvas.create_oval( (x + action_positions[action][0] - action_radius ) *scale,
                                             (y + action_positions[action][1] - action_radius ) *scale,
                                             (x + action_positions[action][0] + action_radius ) *scale,
                                             (y + action_positions[action][1] + action_radius ) *scale, 
                                             outline=color_string, fill=color_string )
        if curr_x is not None and curr_y is not None:
            self.canvas.create_oval( (curr_x + 0.5 - curr_radius ) * scale, 
                                     (curr_y + 0.5 - curr_radius ) * scale, 
                                     (curr_x + 0.5 + curr_radius ) * scale, 
                                     (curr_y + 0.5 + curr_radius ) * scale, 
                                     outline="#fff", fill="#fff" )
        self.root.update()

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)


Common_Memory=Memory(capacity=MEMORY_CAPACITY)
memory_data=[]
#try:
with open("memory.csv","r") as file:
    reader=csv.reader(file)
    for row in reader:
        if len(row)!=4:
            pass
        else:
            memory_data.append(row)
Common_Memory.add(memory_data)
#except:
#    print("no memory yet")
for o in Common_Memory.sample(10000000) :
    if len(o)!=4:
        print("problem in memory",len(o))
# -- set up the mission -- #
#def run_mission(Common_Memory):
for iteration in range(1000):
    agent = TabQAgent()
    agent_host = MalmoPython.AgentHost()
    agent.memory=Common_Memory
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)
    mission_file = './tutorial_6.xml'
    with open(mission_file, 'r') as f:
        print("Loading mission from %s" % mission_file)
        mission_xml = f.read()
        my_mission = MalmoPython.MissionSpec(mission_xml, True)
        my_mission.requestVideo( 320, 240 )

    # add 20% holes for interest
    for x in range(1,4):
        for z in range(1,13):
            if random.random()<0:
                my_mission.drawBlock( x,45,z,"lava")

    max_retries = 3

    if agent_host.receivedArgument("test"):
        num_repeats = 1
    else:
        num_repeats = 50

    cumulative_rewards = []
    for i in range(num_repeats):

        print('Repeat %d of %d' % ( i+1, num_repeats ))
        
        my_mission_record = MalmoPython.MissionRecordSpec()

        for retry in range(max_retries):
            try:
                agent_host.startMission( my_mission, my_mission_record )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission:",e)
                    exit(1)
                else:
                    time.sleep(2.5)

        print("Waiting for the mission to start", end=' ')
        world_state = agent_host.getWorldState()
        while not world_state.has_mission_begun:
            print(".", end="")
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
            for error in world_state.errors:
                print("Error:",error.text)
        print()

        # -- run the agent in the world -- #
        cumulative_reward = agent.run(agent_host)
        print('Cumulative reward: %d' % cumulative_reward)
        cumulative_rewards += [ cumulative_reward ]

        # -- clean up -- #
        time.sleep(0.5) # (let the Mod reset)

    print("Done.")
    Common_Memory=agent.memory
    agent.brain.model.save("model.model")
    print()
    print("Cumulative rewards for all %d runs:" % num_repeats)
    print(cumulative_rewards)
    #return Common_Memory
    #Common_Memory=run_mission(Common_Memory)
    with open("memory.csv","w") as file:
        writer=csv.writer(file)
        for data in Common_Memory.sample(MEMORY_CAPACITY):
            if len(data)==4:
                writer.writerow(data)
