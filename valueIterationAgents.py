# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from sys import _current_frames
import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        
        self.policy = util.Counter()

        print(iterations)
        first = True
        for i in range(self.iterations):
            print(i)
            states = self.mdp.getStates()
            for state in states:
                actions = self.mdp.getPossibleActions(state)
                if actions is None:
                    self.values[state] = 0
                    continue
                for action in actions:
                    # print(self.getQValue(state, action))
                    q_value = self.getQValue(state, action)
                    if first or q_value > self.values[state]:
                        self.values[state] = q_value
                        first = False
                    # self.policy[state] = max(self.policy[state], q_value)
                if i == 0:
                    first = True

            # self.mdp.getPossibleActions(state)
            # do something
        # Write value iteration code here
        "*** YOUR CODE HERE ***"


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        # get next state
        next_options = self.mdp.getTransitionStatesAndProbs(state, action)
        sum = 0
        for option in next_options:
            nextState, prob = option
            sum += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])

        return sum

        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)
        policy = 0
        for action in actions:
            nextStates = self.mdp.getTransitionStatesAndProbs(state, action)
            # print('x', nextState)
            for next_state in nextStates:
                nextState, prob = next_state
                # if self.mdp.isTerminal(nextState):
                #     policy = max(policy, 0)#prob * (self.mdp.getReward(state, action, nextState)))
                # else:
                policy = max(policy, prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState]))

        return policy#self.policy[state]#self.policy[state]
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
