import socket
import socketserver
import numpy as np
from matplotlib import pyplot as plt
import http.server
import socketserver
import pandas as pd
import time
from aiohttp import web
import json
import random

class env_state():
    state = 1

    def __init__(self, agent_num):
        """ State represented as the time_slots request_on has been received
            + The time-slot predicted price forecast for the day"""
        self.state = np.array((1, agent_num*3 + 3))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class MGridEnv():
    metadata = {'render.modes': ['human']}
    time_slot = 0
    actions_sent = [ ]
    agents_sent = 0
    agents_num = 0
    aid_index = 0
    state = ""
    model_schedule = []

    def __init__(self,agent_num, mfd = 4):
        # Load in the workbook
        self.mf_distance = mfd
        self.agents_num = agent_num
        self.actions_sent = [0 for i in range(agent_num)]
        self.power_hist = [ [] for i in range(96)  ]

        """ State for agent is 
        i {time slots sice request} 
        i+1 {requested on by user}
        i+2 {load imposed to the system}"""
        self.state = np.zeros((1, agent_num * 3 + 3))
        print("Shape: ",self.state)

        load_df = pd.read_excel('files/power_production_test.xlsx', index_col=None, header=0)
        appliance_df = pd.read_excel('files/appliance_load_req.xlsx', index_col=None, header=0)

        self.state[0,-3] = load_df.iloc[self.time_slot,2]
        #plt.axis([0, 96, 0, 1000])
        #plt.ion()
        #plt.show()
        print(appliance_df.iloc[0:self.agents_num,1].as_matrix())
        print(self.state[0,2:-3:3])
        self.state[0,2:-3:3] = appliance_df.iloc[0:self.agents_num,1].as_matrix()

        pass

    def _step(self,aid ,action):
        self.actions_sent[aid] = 1

        """Action received 1 if we want to grant the load request
        if this is true the mandatory open time slots are decreased by one in each slot
        when the timer runs out the load request state resets"""


        """If action received is 0 and the load was requested to be ON, 
        the delay counter should increase by one """
        #if action[0] == 12345:
        #    self._timestep_ready()
        # We can tweak this to get th id of the agent via the action
        """

        Parameters
        ----------
        action :

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        self._take_action(action)
        self.status = self._step()
        # Reward is returned in the end of the sim
        reward = self._get_reward()
        ob = self._getState()


        episode_over = self.time_slot == 96 #self.status != hfo_py.IN_GAME
        #We should return only the partial observations

        #return ob, reward, episode_over, {}
        if self._timestep_ready:
            return ob, reward, episode_over, {}

        #else:
        #    return False


        if episode_over:
            return ob, reward, episode_over, {}
        else:
            l_mand_open = list(self.state[0, 0:-3:3])
            l_delay = list(self.state[0, 0:-3:3])
            l_power = list(self.state[0, 0:-3:3])

            #for m, d, p in zip(l_mand_open, l_delay, l_power):
            #    if m < 1:



    def _reset(self):
        l_mand_open = list(self.state[0, 0:-3:3])
        l_delay = list(self.state[0, 0:-3:3])
        l_power = list(self.state[0, 0:-3:3])
        start_slot, mandatory_open , self.model_schedule= [], [], []
        for a in range(self.agents_num):
            start_slot.append(random.randrange(0,89))
            mandatory_open.append(random.randrange(1, 6))
            self.model_schedule.append([start_slot[a], mandatory_open[a]])

        self.actions_sent = [0 for i in range(self.agents_num)]
        self.time_slot = 0
        self.agents_sent = 0

    def _get_state(self,aid = None):

        return self.state


    def _render(self, mode='human', close=False):
        plt.plot(self.status, y)
        plt.draw()
        plt.pause(0.001)


    def _take_action(self, aid , action):
        """ This should define whether or not an agent turns on a requested load
         if the load is not requested on """
        self.agents_sent += 1
        if action == 1:
            if self.state[0,3*aid + 1] > 0:
                self.actions_sent[aid] = 1
                self.state[aid*3] -= 1
                self.state[aid*3 +1] = 0
                return 0
            else:
                return -10000
        return 0


    def _is_terminal(self):
        if self.time_slot >= 96:
            self._reset()
            return True
        else:
            return False

    def _teminal_reward(self,aid):
        pass
        return 0.0

    def _check_reward_type(self,aid):
        if self._timestep_ready():
            if self._is_terminal():
                reward = self._terminal_reward(aid= aid)
            else:
                reward = self._get_reward(aid= aid)
            return reward
        else:
            return False

    def _get_reward(self,aid):
        """ Rewards should be calculated on a per-agent and aggregate basis """
        # We can calculate all the rewards each time
        # per agent reward
        delay_ts = self.state[0,aid * 3 + 1]
        #print(self.state.shape)
        l_mand_open = list(self.state[0,0:-3:3])
        l_delay = list(self.state[0,0:-3:3])
        l_power = list(self.state[0,0:-3:3])

        # global reward
        power_aggregate = 0.0
        power_baseline = 0.0
        agent_penalty = 0.0
        for m, d, p in zip(l_mand_open, l_delay,l_power):
            if m != 0:
                if d != 0:
                    power_aggregate += p
                else:
                    power_baseline += p

        if l_delay[aid]!=0:
            agent_penalty = l_delay[aid] ** 2

        self.power_hist[self.time_slot] = power_aggregate
        #if self._is_terminal():
        #    self._teminal_reward(aid)

        return power_aggregate - agent_penalty


    def _timestep_ready(self):
        #print(self.state)
        #print(self.state[0, 0:-3:3])
        #print(self.state[0, 1:-3:3])
        #print(self.state[0, 2:-3:3])
        if self.agents_sent >= self.agents_num:
            self.time_slot += 1
            for a, ts_mop in enumerate(zip(self.model_schedule)):
                if ts_mop[0] == self.time_slot:
                    self.state[a*3] = ts_mop[1]

            print("Time_slot: " + str(self.time_slot))
            self.actions_sent = [0 for i in range(len(self.actions_sent))]

            return True
        else:
            return False


async def handle(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Hello, " + name
    return web.Response(text=text)

async def _timestep_ready(request):

    #print (request)
    aid = int(request.match_info.get('aid'))
    tsr = env._timestep_ready()
    if tsr == True:
        print("Time Slot: ",env.time_slot)
        res = {"ready" : tsr, "done": env._is_terminal(),"curr_ts": env.time_slot, "reward": env._get_reward(aid),
               "environment": np.array2string(env.state[0,:], max_line_width=15000)[2:-2]}
    else:
        res = {"ready": tsr, "environment": None}

    return web.json_response(res)

async def _step(request):
    aid = int(request.match_info.get('aid'))
    action = int(request.match_info.get('action'))
    print("Agent: ", aid, "  Action: ", action)
    #response = env._step(aid = aid, action =  action)
    res = {"action":action , "aid":aid }#np.array_str(action)}

    return web.json_response(res)


async def _take_action(request):
    aid = int(request.match_info.get('aid'))
    action = int(request.match_info.get('action'))
    #print("Agent: ", aid, "  Action: ", action)

    act_res = float(env._take_action(aid,action))
    #response = env._step(aid = aid, action =  action)
    res = {"aid":aid , "reward": act_res}#np.array_str(action)}

    return web.json_response(res)

async def _get_reward(request):
    name = request.match_info.get('name', "Anonymous")
    text = "Hello, " + name

    return web.Response(text=text)

async def _observe(request):
    aid = int(request.match_info.get('aid'))
    res = {"environment": np.array2string(env.state[0,:], max_line_width=15000)[2:-2]}
    #print(res)
    return web.json_response(res)

async def _get_aid(request):
    new_aid = env.aid_index
    env.aid_index += 1
    res = {"your_aid": new_aid}
    return web.json_response(res)


if __name__ == '__main__':

    env = MGridEnv(agent_num=4 , mfd=5)
    app = web.Application()
    app.add_routes([web.get('/', handle),
                    web.get('/ts_ready/{aid}', _timestep_ready),
                    web.get('/step/{aid}/{action}', _step),
                    web.get('/make_action/{aid}/{action}', _take_action),
                    web.get('/observe/{aid}', _observe),
                    web.get('/get_aid', _get_aid),
                    web.get('/{name}', handle)])

    web.run_app(app, host="127.0.0.1", port=1994)

