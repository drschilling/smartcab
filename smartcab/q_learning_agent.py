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
        self.temp_q_table = {}
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.num_trials = 1  # number of trials
        self.alpha = 0.0  # learning rate initialization
        self.epsilon = 0.0  # the probability of an event should occur
        self.actions = [None, 'forward', 'left', 'right']  # set of possible actions
        self.waypoints = ['forward', 'right', 'left']  # set of possible waypoints
        self.lights = ['red', 'green']  # set of possible lights
        self.sum_time_left = 0  # sum of the time left till the agent reaches its deadline
        self.n_penalties = 0  # number of penalties incurred
        self.destination_reached = False  # number of times that our smartcab reach its destination
        self.q_table = self.create_q_table(self.actions, self.waypoints, self.lights)  # training scenario

    def reset(self, destination=None):
        self.planner.route_to(destination)
        self.num_trials += 1  # increment the number of trials
        self.write_state_to_csv(self.num_trials, self.alpha, self.epsilon, self.n_penalties, self.destination_reached,
                                self.env.get_sum_time_left())  # creates an output file with metrics every new trip
        self.n_penalties = 0  # reset the number of penalties
        self.destination_reached = False  # reset if the destination is reached
        self.sum_time_left = 0  # reset the time left

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        self.state = (inputs['light'], inputs['oncoming'], self.next_waypoint)  # create a primary state

        # Updating the current state
        max_q_value = self.q_table[self.state].index(max(self.q_table[self.state]))

        # Selecting an action Epsilon greedy policy that is selecting random actions with uniform distribution
        # from a set of available actions.
        self.epsilon = 1.0 / self.num_trials
        if random.randint(0, 10) < self.epsilon:
            act = random.choice(self.env.valid_actions) # selecting a random action
        else:
            act = self.actions[max_q_value] # selecting a max action

        reward_from_action = self.env.act(self, act) # from the action above we will have a reward

        # if the reward is received as a penalty we count that
        if reward_from_action < 0:
            self.n_penalties += 1

        location = self.env.agent_states[self]["location"]
        destination = self.env.agent_states[self]["destination"]

        # if the agent reaches its destination we save that information
        if location == destination:
            self.destination_reached = True

        self.alpha = 1.0 / self.num_trials # learning rate decay
        next_inputs = self.env.sense(self) # a new set of inputs
        new_waypoints = self.planner.next_waypoint() # a new set of waypoints

        # from the policy we generate a new state from the next set of inputs
        next_state = (next_inputs['light'], next_inputs['oncoming'], new_waypoints)

        # get maximum Q value for this next state based on all possible actions and learning rate
        self.q_learning(act, self.alpha, reward_from_action, next_state)

    def create_q_table(self, actions, waypoints, lights):
        for i in lights:
            for j in actions:
                for k in waypoints:
                    self.temp_q_table[(i, j, k)] = [1] * len(actions)
        return self.temp_q_table

    def q_learning(self, act, alpha, reward_from_action, next_state):
        self.q_table[self.state][self.actions.index(act)] = (1 - alpha) * self.q_table[self.state][
            self.actions.index(act)] + (alpha * (reward_from_action + max(self.q_table[next_state])))

    @staticmethod
    def write_state_to_csv(num_trials, alpha, epsilon, n_penalties, destination_reached, time_left):
        output_file = open('results/q_learning_output.csv', 'a')
        writer = csv.writer(output_file)
        writer.writerow((num_trials, alpha, epsilon, n_penalties, destination_reached, time_left))
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
