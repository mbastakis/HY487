# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    """***CODE CSD4406***"""
    # Prints I tested for help
    # print(problem)
    # print("Start:", problem.getStartState())
    # print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    # print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    # t = problem.getStartState()
    # print(t)
    # temp = problem.getSuccessors(t)
    # print(temp[0][0])
    # print("Next,", problem.getSuccessors(temp[0][0]))

    from util import Stack
    stack = Stack()
    visitedCoords = []

    # If startingCoords is the same with goalState
    startingCoords = problem.getStartState()
    if problem.isGoalState(startingCoords):
        return []

    # Push source tuple to stack. tuple = (Current Coords, Path from starting coords)
    stack.push((startingCoords, []))

    while not stack.isEmpty():
        # Get information from current state
        currCoords, currPath = stack.pop()
        # Stack may have duplicate coords, if coords are not visited
        if currCoords not in visitedCoords:
            # Visit current coords
            visitedCoords.append(currCoords)
            # Check if current coords is the goal state
            if problem.isGoalState(currCoords):
                print("Current Path:", currPath)
                print("Length:", len(currPath))
                return currPath
            # Expand currCoords and add to stack every successor
            for successor in problem.getSuccessors(currCoords):
                nextCoords = successor[0]
                moveDirection = successor[1]
                # Find path for new coordinates
                nextPath = currPath + [moveDirection]
                # Push the new coordinates to the stack
                stack.push((nextCoords, nextPath))

    # Stack was empty, search failed couldn't find solution
    return []


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    """
    *** CODE CSD4406***
    Breadth first search has the exact same algorithm, the only difference is that 
    instead of a stack for the fringe we use a queue
    """

    from util import Queue
    queue = Queue()
    visitedCoords = []

    # If startingCoords is the same with goalState
    startingCoords = problem.getStartState()
    if problem.isGoalState(startingCoords):
        return []

    # Push source tuple to queue. tuple = (Current coords, Path from starting coords)
    queue.push((startingCoords, []))

    while not queue.isEmpty():
        # Get information from current state
        currCoords, currPath = queue.pop()
        # Queue may have duplicate coords, if coords are not visited
        if currCoords not in visitedCoords:
            # Visit current coords
            visitedCoords.append(currCoords)
            # Check if current coords is the goal state
            if problem.isGoalState(currCoords):
                print("Current Path:", currPath)
                print("Length:", len(currPath))
                return currPath
            # Expand currCoords and add to queue every successor
            for successor in problem.getSuccessors(currCoords):
                nextCoords = successor[0]
                moveDirection = successor[1]
                # Find path for new coordinates
                nextPath = currPath + [moveDirection]
                # Push the new coordinates to the queue
                queue.push((nextCoords, nextPath))

    # Queue was empty, search failed couldn't find solution
    return []


def uniformCostSearch(problem):
    "Search the node of least total cost first. "
    "*** YOUR CODE HERE ***"

    util.raiseNotDefined()


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    """
    ***CODE CSD4406***
    Again the code is similar with DFS and BFS code but instead we use a
    priority queue and we assign costs to each coordinate
    """
    from util import PriorityQueue
    # A list of coordinates which have been visited,
    # but neighboring coordinates still haven't
    openList = PriorityQueue()
    # A list of coordinates which have been visited,
    # and neighboring coordinates have also been expanded
    closedList = []

    # If startingCoords is the same with goalState
    startingCoords = problem.getStartState()
    if problem.isGoalState(startingCoords):
        return []

    startNode = {
        "coords": startingCoords,
        "path": [],
        "cost": 0
    }
    # Push starting node with priority 0
    openList.push(startNode, 0)

    while not openList.isEmpty():
        currentNode = openList.pop()
        # Priority Queue may have duplicate cords, if state is not visited
        if currentNode["coords"] not in closedList:
            # Visit current coords
            closedList.append(currentNode["coords"])
            # Check if current coords is the goal state
            if problem.isGoalState(currentNode["coords"]):
                print("Current Path:", currentNode["path"])
                print("Length:", len(currentNode["path"]))
                return currentNode["path"]
            # Expand currCoords and add to priority queue every successor
            for successor in problem.getSuccessors(currentNode["coords"]):
                # Create and push node for current successor
                node = {
                    "coords": successor[0],
                    "path": currentNode["path"] + [successor[1]],
                    "cost": currentNode["cost"] + successor[2]
                }
                # Find f cost using g and h cost -> f(n) = g(n) + h(n)
                f_cost = node["cost"] + heuristic(node["coords"], problem)
                # Push the new coordinates to the queue
                openList.push(node, f_cost)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
