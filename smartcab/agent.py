import random
import numpy as np
import pandas as pd
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.temp_q_table = {}
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.numTrials = 0
        self.epsilon = 1.0
        self.gamma = 0.2
        self.actions = [None, 'forward', 'left', 'right']
        self.waypoints = ['forward', 'right', 'left']
        self.lights = ['red', 'green']
        self.q_table = self.create_q_table(self.actions, self.waypoints, self.lights)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.numTrials += 1
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state = (inputs['light'], inputs['oncoming'], self.next_waypoint)

        # TODO: Update state
        max_q_value = self.q_table[self.state].index(max(self.q_table[self.state]))

        # TODO: Select action according to your policy
        if random.randint(0, 10) < self.epsilon:
            act = random.choice(self.env.valid_actions)
        else:
            act = self.actions[max_q_value]
        reward_from_action = self.env.act(self, act)

        # TODO: Learn policy based on state, action, reward
        alpha = 1 / np.log(self.numTrials + 2)

        nxt_inputs = self.env.sense(self)
        new_waypoints = self.planner.next_waypoint()
        nxt_state = (nxt_inputs['light'], nxt_inputs['oncoming'], new_waypoints)

        self.q_table[self.state][self.actions.index(act)] = (1 - alpha) * self.q_table[self.state][
            self.actions.index(act)] + (alpha * (reward_from_action + self.gamma * max(self.q_table[nxt_state])))

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs,
                                                                                                    act,
                                                                                                    reward_from_action)  # [debug]

    def create_q_table(self, actions, waypoints, lights):
        for i in lights:
            for j in actions:
                for k in waypoints:
                    self.temp_q_table[(i, j, k)] = [1] * len(actions)
        return self.temp_q_table


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.5, display=True)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
    # results = []
    # for i in range(100):
    #    sim_results = run()
    #     results.append(run())
    # df_results = pd.DataFrame(results)
    # df_results.columns = ['reward_sum', 'disc_reward_sum', 'n_dest_reached',
    #                       'last_dest_fail', 'sum_time_left', 'n_penalties',
    #                       'last_penalty', 'len_qvals']
    # df_results.to_csv('original_agent_results.csv')
