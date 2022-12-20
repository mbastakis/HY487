# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** CSD4406 CODE ***"
        import sys

        # Win condition        
        if successorGameState.isWin():
            return sys.float_info.max
        # Pacman movements
        if currentGameState.getPacmanPosition() == successorGameState.getPacmanPosition():
          return sys.float_info.min

        # Final score to be returned
        score = 0

        # Ghost Calculations
        currGhostDist = []
        newGhostDist = []
        # New ghost distances
        for ghostPos in successorGameState.getGhostPositions():
          currGhostDist.append(manhattanDistance(newPos, ghostPos))
        # Current ghost distances
        for ghostPos in currentGameState.getGhostPositions():
          newGhostDist.append(manhattanDistance(newPos, ghostPos))
        score += -500 if min(currGhostDist) > min(newGhostDist) else 1000

        # Food Calculations
        foodDist = []
        for foodPos in newFood.asList():
          foodDist.append(manhattanDistance(newPos, foodPos))
        score += float(1/min(foodDist)) * 10
        score -= len(newFood.asList())

        return successorGameState.getScore() + score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.
      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  pacmanAgentIndex = 0

  def getAction(self, gameState):
      """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
          Returns a list of legal actions for an agent
          agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action):
          Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
          Returns the total number of agents in the game
      """
      "*** CSD4406 CODE ***"
      return self.maxValue(gameState, self.pacmanAgentIndex, 0)[0]

  def minimax(self, state, agentIndex, depth):
    if depth == self.depth * state.getNumAgents() or state.isLose() or state.isWin():
        return self.evaluationFunction(state)
    if agentIndex == self.pacmanAgentIndex:
        return self.maxValue(state, agentIndex, depth)[1]
    else:
        return self.minValue(state, agentIndex, depth)[1]

  def maxValue(self, state, agentIndex, depth):
    import sys
    maxAction = ("max", -sys.float_info.max)

    newDepth = depth + 1
    newAgentIndex = (newDepth) % state.getNumAgents()

    for action in state.getLegalActions(agentIndex):
      newState = state.generateSuccessor(agentIndex, action)
      newAction = (action, self.minimax(newState, newAgentIndex, newDepth))
      if maxAction[1] < newAction[1]:
        maxAction = newAction
    return maxAction

  def minValue(self, state, agentIndex, depth):
    import sys
    minAction = ("min", sys.float_info.max)

    newDepth = depth + 1
    newAgentIndex = newDepth % state.getNumAgents()

    for action in state.getLegalActions(agentIndex):
      newState = state.generateSuccessor(agentIndex,action)
      newAction = (action,self.minimax(newState, newAgentIndex, newDepth))
      if minAction[1] > newAction[1]:
        minAction = newAction
    return minAction


class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """
  pacmanAgentIndex = 0

  def getAction(self, gameState):
    """
    Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** CSD4406 CODE ***"
    import sys
    
    self.gameState = gameState
    legalActions = gameState.getLegalActions(self.pacmanAgentIndex)
    alpha = -sys.float_info.max
    beta = sys.float_info.max

    actions = []
    for legalAction in legalActions:
      newState = gameState.generateSuccessor(0, legalAction)
      value = self.minValue(newState, 1, 1, alpha, beta)
      actions.append((legalAction, value))

      if value > beta:
        return legalAction
      elif value > alpha:
        alpha = value

    maxAction = max(actions, key=lambda x: x[1])[0]
    return maxAction
    
  def minValue(self, state, agentIndex, depth, alpha, beta):
    import sys

    legalActions = state.getLegalActions(agentIndex)
    if len(legalActions) == 0:
      return self.evaluationFunction(state)

    minValue = sys.float_info.max
    
    if agentIndex == self.gameState.getNumAgents() - 1:
      for legalAction in legalActions:
        maxValue = self.maxValue(state.generateSuccessor(agentIndex, legalAction), agentIndex,  depth, alpha, beta)
        minValue =  min(minValue, maxValue)
        if minValue < alpha:
          return minValue
        elif minValue < beta:
          beta = minValue
    else:
      for legalAction in legalActions:
        successorMinValue = self.minValue(state.generateSuccessor(agentIndex, legalAction), agentIndex + 1, depth, alpha, beta)
        minValue =  min(minValue, successorMinValue)
        if minValue < alpha:
          return minValue
        elif minValue < beta:
          beta = minValue

    return minValue

  def maxValue(self, state, agentIndex, depth, alpha, beta):
    import sys

    legalActions = state.getLegalActions(self.pacmanAgentIndex)
    if len(legalActions) == 0  or depth == self.depth:
      return self.evaluationFunction(state)

    maxValue = -sys.float_info.max

    for legalAction in legalActions:
      minValue = self.minValue(state.generateSuccessor(self.pacmanAgentIndex, legalAction), self.pacmanAgentIndex + 1, depth + 1, alpha, beta) 
      maxValue = max(maxValue, minValue)
      if maxValue > beta:
        return maxValue
      elif maxValue > alpha:
        alpha = maxValue

    return maxValue    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    pacmanAgentIndex = 0

    def maxValue(self, state, depth):
      import sys

      legalActions = state.getLegalActions(self.pacmanAgentIndex)
      if len(legalActions) == 0 or depth == self.depth:
        return self.evaluationFunction(state)

      maxExpValue = sys.float_info.min
      for legalAction in legalActions:
        currentValue = self.expValue(state.generateSuccessor(0, legalAction), 0 + 1, depth + 1)
        if maxExpValue <= currentValue:
          maxExpValue = currentValue
      
      return maxExpValue

    def expValue(self, state, agentIndex, depth):
      legalActions = state.getLegalActions(agentIndex)
      if len(legalActions) == 0:
        return self.evaluationFunction(state)

      maxValue = 0
      for legalAction in legalActions:
        newState = state.generateSuccessor(agentIndex, legalAction)

        if agentIndex == state.getNumAgents() - 1:
          maxValue += self.maxValue(newState, depth) / len(legalActions)
        else:
          maxValue += self.expValue(newState, agentIndex + 1, depth) / len(legalActions)
      return maxValue

    def getAction(self, gameState):
      """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        The expectimax function returns a tuple of (actions,
      """
      "*** CSD4406 HERE ***"
      
      return max(gameState.getLegalActions(), \
                key=lambda action: self.expValue(gameState.generateSuccessor(0, action), 1, 1))
      

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
      Evaluate state by  :
            * closest food
            * food left
            * capsules left
            * distance to ghost
    """
    "*** YOUR CODE HERE ***"
    import sys
    score = currentGameState.getScore()
    pacmanPos = currentGameState.getPacmanPosition()
    
    for foodPos in currentGameState.getFood().asList():
      foodDist = manhattanDistance(pacmanPos, foodPos)
      score += (1/float(foodDist))

    for ghostPos in currentGameState.getGhostPositions():
      ghostDist = manhattanDistance(pacmanPos, ghostPos)
      if ghostDist == 0:
          return sys.float_info.min
      if(ghostDist < 3):
          score += 10 * (1 / float(ghostDist))  
      else:
          score += (1 / float(ghostDist)) 

    return score


# Abbreviation
better = betterEvaluationFunction
