import socket
import socketserver
import numpy as np
from matplotlib import pyplot as plt
import http.server
import socketserver

import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import urllib.request
contents = urllib.request.urlopen("http://example.com/foo/bar").read()

class env_state():
    state = 1

    def __init__(self, agent_num):
        """ State represented as the time_slots request_on has been received
            + The time-slot predicted price forecast for the day"""
        self.state = np.array((1, agent_num + 3))


class MGridEnv():
    metadata = {'render.modes': ['human']}
    time_slot = 0
    actions_sent = [ ]
    agents_num = 0



    def __init__(self,agent_num, mfd = 4):
        # Load in the workbook
        self.mf_distance = mfd
        self.agents_num = agent_num
        self.actions_sent = [0 for i in range(agent_num)]

        load_df = pd.read_excel('/Documents/thesis_test_load_req.xlsx', index_col=None, header=0)


        plt.axis([0, 96, 0, 1000])
        plt.ion()
        plt.show()

        pass

    def _step(self, action):
        self.actions_sent[action[0]] = 1

        if action[0] == 12345:
            self._timestep_ready()
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
        self._take_action(action[1:])
        self.status = self.env.step()
        # Reward is returned in the end of the sim
        reward = self._get_reward()
        ob = self.env.getState()


        episode_over = self.time_slot == 96 #self.status != hfo_py.IN_GAME
        #We should return only the partial observations

        #return ob, reward, episode_over, {}
        if self._timestep_ready:
            return ob, reward, episode_over, {}
        else:
            return False

    def _reset(self):
        self.actions_sent = [0 for i in range(self.agents_num)]


    def _render(self, mode='human', close=False):
        plt.plot(self.status, y)
        plt.draw()
        plt.pause(0.001)


    def _take_action(self, action):
        """ This should define whether or not an agent turns on a requested load
         if the load is not requested on """

        pass

    def _get_reward(self,agent_id):
        """ Rewards should be calculated on a per-agent and aggregate basis """
        # We can calculate all the rewards each time
        """ Reward is given for XY. """
        if self.status == FOOBAR:
            return 1
        elif self.status == ABC:
            return self.somestate ** 2
        else:
            return 0

    def _timestep_ready(self):
        if sum (self.actions_sent) == self.agents_num:
            self.time_slot += 1
            return True #"ready"
        else:
            return False


HOST_NAME = 'localhost'
PORT_NUMBER = 9000

class MyHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        paths = {
            '/foo': {'status': 200},
            '/bar': {'status': 302},
            '/baz': {'status': 404},
            '/qux': {'status': 500}
        }

        if self.path in paths:
            self.respond(paths[self.path])
        else:
            self.respond({'status': 500})

    def handle_http(self, status_code, path):
        self.send_response(status_code)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        content = '''
        <html><head><title>Title goes here.</title></head>
        <body><p>This is a test.</p>
        <p>You accessed path: {}</p>
        </body></html>
        '''.format(path)
        return bytes(content, 'UTF-8')

    def respond(self, opts):
        response = self.handle_http(opts['status'], self.path)
        self.wfile.write(response)

if __name__ == '__main__':
    server_class = HTTPServer
    httpd = server_class((HOST_NAME, PORT_NUMBER), MyHandler)
    print(time.asctime(), 'Server Starts - %s:%s' % (HOST_NAME, PORT_NUMBER))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()
    print(time.asctime(), 'Server Stops - %s:%s' % (HOST_NAME, PORT_NUMBER))