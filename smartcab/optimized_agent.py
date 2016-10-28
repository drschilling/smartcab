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
        self.num_trials = 1
        self.gamma = 0.2
        self.alpha = 0.0
        self.epsilon = 0.0
        self.actions = [None, 'forward', 'left', 'right']
        self.waypoints = ['forward', 'right', 'left']
        self.lights = ['red', 'green']
        self.n_penalties = 0  # number of penalties incurred
        self.destination_reached = False
        self.q_table = self.create_q_table(self.actions, self.waypoints, self.lights)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.num_trials += 1
        self.write_state_to_csv(self.alpha, self.epsilon, self.num_trials, self.n_penalties, self.destination_reached)
        self.n_penalties = 0
        self.destination_reached = False
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
        self.epsilon = 1.0 / self.num_trials
        if random.randint(0, 10) < self.epsilon:
            act = random.choice(self.env.valid_actions)
        else:
            act = self.actions[max_q_value]

        reward_from_action = self.env.act(self, act)

        if reward_from_action < 0:
            self.n_penalties += 1

        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]

        if location == destination:
            self.destination_reached = True

        # TODO: Learn policy based on state, action, reward
        self.alpha = 1 / np.log(self.num_trials + 2)
        next_inputs = self.env.sense(self)
        new_waypoints = self.planner.next_waypoint()
        next_state = (next_inputs['light'], next_inputs['oncoming'], new_waypoints)

        self.update_q(act, self.alpha, reward_from_action, next_state)

    def create_q_table(self, actions, waypoints, lights):
        for i in lights:
            for j in actions:
                for k in waypoints:
                    self.temp_q_table[(i, j, k)] = [1] * len(actions)
        return self.temp_q_table

    def update_q(self, act, alpha, reward_from_action, next_state):
        self.q_table[self.state][self.actions.index(act)] = (1 - alpha) * self.q_table[self.state][
            self.actions.index(act)] + (alpha * (reward_from_action + self.gamma * max(self.q_table[next_state])))

    @staticmethod
    def write_state_to_csv(alpha, epsilon, num_trials, n_penalties, destination_reached):
        output_file = open('optimized_output.csv', 'a')
        writer = csv.writer(output_file)
        writer.writerow((alpha, epsilon, num_trials, n_penalties, destination_reached))
        output_file.close()


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.2, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()