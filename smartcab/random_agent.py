import csv
import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(
            env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.number_of_trials = 1  # number of trials
        self.color = 'red'  # override color
        self.sum_time_left = 0  # total sum of the time left till one trial reaches the deadline
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.actions = ['forward', 'left', 'right', None]  # set of possible actions
        self.number_of_penalties = 0  # number of penalties incurred
        self.destination_reached = False  # number of times that our smartcab reach its destination

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.number_of_trials += 1  # increment the number of trials
        self.write_state_to_csv(self.number_of_trials, self.number_of_penalties, self.destination_reached,
                                self.env.get_sum_time_left()) # creates an output file with metrics every new trip
        self.number_of_penalties = 0  # reset the number of penalties
        self.destination_reached = False  # reset if the destination is reached
        self.sum_time_left = 0  # reset the time left

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator

        # random choice of actions
        act = random.choice(self.actions)

        reward = self.env.act(self, act)
        if reward < 0:
            self.number_of_penalties += 1

        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]

        if location == destination:
            self.destination_reached = True

    @staticmethod
    def write_state_to_csv(number_of_trials, number_of_penalties, destination_reached, time_left):
        output_file = open('results/random_output.csv', 'a')
        writer = csv.writer(output_file)
        writer.writerow((number_of_trials, number_of_penalties, destination_reached, time_left))
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
