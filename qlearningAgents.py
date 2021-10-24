# qlearningAgents.py
# ------------------
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


from os import X_OK
from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.q_values = dict()#util.Counter()

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        if state in self.q_values:
          if action in self.q_values[state]:
            return self.q_values[state][action]
        return 0
        # return self.q_values[state][action]
        util.raiseNotDefined()


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if actions is None or len(actions) == 0:
          return 0
        max_val = self.getQValue(state, actions[0])
        for action in actions[1:]:
          val = self.getQValue(state, action)
          if val > max_val:
            max_val = val
        return max_val
        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)
        if actions is None or len(actions) == 0:
          return None
        max_val = self.getQValue(state, actions[0])
        max_action = actions[0]
        for action in actions[1:]:
          val = self.getQValue(state, action)
          if val > max_val:
            max_val = val
            max_action = action
        return max_action

        util.raiseNotDefined()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        if util.flipCoin(self.epsilon):
          return random.choice(self.getLegalActions(state))
        else:
          return self.computeActionFromQValues(state)
        # legalActions = self.getLegalActions(state)
        # action = None
        # "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        # return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        value = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))  
        if state in self.q_values:
          self.q_values[state][action] = value
        else:
          self.q_values[state] = {action : value}
        # self.q_values[state][action] = (1 - self.alpha) * self.q_values[state][action] + self.alpha * (reward + self.discount * self.computeValueFromQValues(nextState))  
        "*** YOUR CODE HERE ***"

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        product = 0
        # for i in range(len(self.featExtractor)):
        # for i in self.featExtractor.sortedKeys():
        weight = self.weights[(state, action)]
        if weight == 0:
          return 0
        print("w", weight)

        feature = self.featExtractor.getFeatures(state, action)
        print("f", feature)

        for key in weight.keys():
            product += (weight[key] * feature[key])
            # print(product)
        # for key in self.featExtractor.getFe
        # features = self.featExtractor.getFeatures(state, action)
        # product = 0
        # product = weight * features[state]
        # for i in range(len(weight)):
        #   product += weight[i] * features[i]
        # product = self.weights[(state, action)] * self.featExtractor.getFeatures(state, action)
        return product
        # util.raiseNotDefined()

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        print(action)
        # for i in self.featExtractor.sortedKeys():
        update = self.alpha * (reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)) #* self.featExtractor.getFeatures(state, action)
        feature = self.featExtractor.getFeatures(state, action)
        for key in feature.keys():
          feature[key] *= update
        # print("f", feature.keys())

        # if type(feature[state]) is int:
        # feature *= update
        # else:
        #   for i in range(len(feature)):
        #     feature[i] *= update
        # if self.weights[(state, action)] == 0:
        for key in feature.keys():
          if type(self.weights[(state, action)]) is not dict:
            self.weights[(state, action)] = {key: feature[key]}
          elif key in self.weights[(state, action)]:
            self.weights[(state, action)][key] += feature[key]
          else:
          #   print(self.weights[(state, action)], key)
          #   val = 0
            # for value in self.weights[(state, action)].values():
            #   val = value
            print(action, key)
            self.weights[(state, action)][key] = feature[key]
        # self.weights[(state, action)] = feature
        # else:
        #   for i in range(len(feature)):
        #     self.weights[(state, action)][i] += feature[i]  
        # self.weights[(state, action)] = update
        # print(self.weights)
        # util.raiseNotDefined()

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
          print(self.weights)
            # you might want to print your weights here for debugging
            # "*** YOUR CODE HERE ***"
            # pass
