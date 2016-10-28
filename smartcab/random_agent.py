import csv
import random
import numpy as np
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
        self.actions = [None, 'forward', 'left', 'right']
        self.n_penalties = 0  # number of penalties incurred
        self.destination_reached = False
        self.qvals = {}  # mapping (state, action) to q-values

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.write_state_to_csv(self.n_penalties, self.destination_reached)
        self.n_penalties = 0
        self.destination_reached = False
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        self.state = (inputs['light'], inputs['oncoming'], self.next_waypoint)

        act = random.choice(self.env.valid_actions)

        reward = self.env.act(self, act)
        if reward < 0:
            self.n_penalties += 1

        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]

        if location == destination:
            self.destination_reached = True

        self.qvals[(self.state, act, reward)] = 0

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}, n_penalties = {}".format(
            deadline, inputs,
            act,
            reward,
            self.n_penalties)  # [debug]

    @staticmethod
    def write_state_to_csv(n_penalties, destination_reached):
        output_file = open('random_output.csv', 'a')
        writer = csv.writer(output_file)
        writer.writerow((n_penalties, destination_reached))
        output_file.close()


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.2,
                    display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
