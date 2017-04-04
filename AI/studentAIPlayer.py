import random
import sys
import operator
import numpy as np
sys.path.append("..")  #so other modules can be found in parent dir
from Player import *
from Constants import *
from Construction import CONSTR_STATS
from Ant import UNIT_STATS
from Move import Move
from GameState import *
from AIPlayerUtils import *


##
#AIPlayer
#Description: The responsbility of this class is to interact with the game by
#deciding a valid move based on a given game state. This class has methods that
#will be implemented by students in Dr. Nuxoll's AI course.
#
#Variables:
#   playerId - The id of the player.
##
class AIPlayer(Player):

    #__init__
    #Description: Creates a new Player
    #
    #Parameters:
    #   inputPlayerId - The id to give the new player (int)
    ##
    def __init__(self, inputPlayerId):
        super(AIPlayer,self).__init__(inputPlayerId, "MiniMatt")
        #limit of how deep function should recurse
        self.depthLimit = 3
        self.me = inputPlayerId
        
        

    ##
    #getPlacement
    #
    #Description: called during setup phase for each Construction that
    #   must be placed by the player.  These items are: 1 Anthill on
    #   the player's side; 1 tunnel on player's side; 9 grass on the
    #   player's side; and 2 food on the enemy's side.
    #
    #Parameters:
    #   construction - the Construction to be placed.
    #   currentState - the state of the game at this point in time.
    #
    #Return: The coordinates of where the construction is to be placed
    ##
    def getPlacement(self, currentState):
        numToPlace = 0
        #implemented by students to return their next move
        if currentState.phase == SETUP_PHASE_1:    #stuff on my side
            numToPlace = 11
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on your side of the board
                    y = random.randint(0, 3)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        elif currentState.phase == SETUP_PHASE_2:   #stuff on foe's side
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                move = None
                while move == None:
                    #Choose any x location
                    x = random.randint(0, 9)
                    #Choose any y location on enemy side of the board
                    y = random.randint(6, 9)
                    #Set the move if this space is empty
                    if currentState.board[x][y].constr == None and (x, y) not in moves:
                        move = (x, y)
                        #Just need to make the space non-empty. So I threw whatever I felt like in there.
                        currentState.board[x][y].constr == True
                moves.append(move)
            return moves
        else:
            return [(0, 0)]
    
    ##
    #getMove
    #Description: Gets the next move from the Player.
    #
    #Parameters:
    #   currentState - The state of the current game waiting for the player's move (GameState)
    #
    #Return: The Move to be made
    ##
    def getMove(self, currentState):
        move = self.recurseEval(currentState, 0, 0, 1)
        if move is None:
            return Move(END, None, None)
        return move        
           
    
    ##
    #getAttack
    #Description: Gets the attack to be made from the Player
    #
    #Parameters:
    #   currentState - A clone of the current state (GameState)
    #   attackingAnt - The ant currently making the attack (Ant)
    #   enemyLocation - The Locations of the Enemies that can be attacked (Location[])
    ##
    def getAttack(self, currentState, attackingAnt, enemyLocations):
        #Attack a random enemy.
        return enemyLocations[random.randint(0, len(enemyLocations) - 1)]

    ##
    # recurseEval
    #
    # Description: This is a recursive function that chooses what move
    #               the agent should take based on what move will give
    #               it the best possible state.
    #
    # Parameters: 
    #   currentState - the current state that is being checked
    #   currentDepth - the current depth of the search, used to make
    #                   sure you dont get too deep
    #   
    #   Returns:
    #       This function can return multiple things depnding on the depth
    #       Depth = 0
    #           Will return the best move or None if no possible moves
    #       Depth !=0
    #           Will return the score of the current state back to the calling function
    ##
    def recurseEval(self, currentState, currentDepth, alpha, beta):
    

        #Base Case
        if currentDepth == self.depthLimit:
            return self.getScore(currentState, currentDepth)
 
# get all valid moves
        allMoves = self.listAllRecurseMoves(currentState)
        nextMoves= []
        for i in allMoves:
            if i.moveType == MOVE_ANT:
# remove moves where queen ends up on her anthill
                if i.coordList[0] == currentState.inventories[self.playerId].getQueen().coords:
                    if i.coordList[-1] == currentState.inventories[self.playerId].getAnthill().coords:
                        continue
# remove moves that go outof the territory
                if isPathOkForQueen(i.coordList):
                    nextMoves.append(i)
                    
        #Create our list of Nodes to evaluate
        nextNodes = [dict({'latestMove': move, 'state': getNextStateAdversarial(currentState, move)}) for move in nextMoves] 
        for node in nextNodes:
            node['score'] = float(currentState.inventories[self.playerId].foodCount)/\
            ((currentState.inventories[self.playerId].foodCount)+(currentState.inventories[(self.playerId+1)%2].foodCount))
        bestMove = None
        # if you have met the depth limit
        if currentDepth < self.depthLimit:
            #Sorts nodes and then takes the top 10 scored nodes
            sortedNodes = nextNodes.sort(key=operator.itemgetter('score'))
            sortedNodes = nextNodes[:10]
           
            #MAX
            if currentState.whoseTurn == self.playerId:
# initial max val is 0
                bestVal = 0.0
                for node in sortedNodes:
                    newVal =  self.recurseEval(node['state'], (currentDepth + 1), alpha, beta) 
                    
# if the current score is greater than the best recorded for the state
                    if newVal > bestVal:
# change the best value from that state
                        bestVal = newVal
# change the beta value if the new best value is greater than the current one value
                        if bestVal > alpha:
                            alpha = bestVal
# prune if beta is less than or equal to alpha
                        if beta <= alpha:
                            #print "PRUNE ?" + "alpha = " + str(alpha) + "\tbeta" + str(beta)
                            break
# change the best move from that state
                        bestMove = node['latestMove']
                if currentDepth == 0:
                    #print "RETURN"
                    return bestMove
                #print "BestVal="+str(bestVal)
                return bestVal
            #MIN
            else:
# reverse the order of the sorted nodes to optimize alpha beta pruning for the min player
                sortedNodes.reverse()
# initial max val is 0
                bestVal = 1.0
                for node in sortedNodes:
                    newVal =  self.recurseEval(node['state'], (currentDepth + 1), alpha, beta) 
                    
# if the current score is less than than the best recorded for the state
                    if newVal < bestVal:
                        bestVal = newVal
                        if bestVal < beta:
# change the beta value if the new best value is less than the current minimum value
                            beta = bestVal
# prune if beta is less than or equal to alpha
                        if beta <= alpha:
                            #print "PRUNE ?" + "alpha = " + str(alpha) + "\tbeta" + str(beta)
                            break
# change the best move from that state
                        bestMove = node['latestMove']
                if currentDepth == 0:
                    #print "RETURN"
                    return bestMove
                #print "BestVal="+str(bestVal)
                return bestVal
    ##
    # getScore
    #       Description: Takes a state and evaluates it from a score of 0-1
    #               This functions takes certian critera into account and weighs
    #               it.
    #
    #       Params: currentState
    #           The state that is being evaluated
    #
    #       Return:
    #           Returns the score of the state, a double between 0-1

    def getScore(self, state, depth):
    
        ids = [(self.playerId + 1) % 2, self.playerId] # opponent = 0, player = 1, return i
        
        scores = [100, 100]
        scores[state.whoseTurn] += 10*(self.depthLimit-depth)
        #opponent first
        for p in range(0,2):
            scores[p] += 500 * state.inventories[ids[p]].foodCount
            for a in state.inventories[ids[p]].ants:
                scores[p] += 20
                #evaluate workers on their food gathering performance
                if a.type == WORKER:
                    if a.carrying:
                        scores[p] -= 1 * approxDist(a.coords, state.inventories[ids[p]].getAnthill().coords)
                        scores[p] += 200
                    if not a.carrying:
                        scores[p] -= 1 * approxDist(a.coords, getConstrList(state, None, (FOOD,))[0].coords) #food location

        fudge = abs(min(scores)) + 1

        scores[0] += fudge
        scores[1] += fudge
        return scores[1]/float(sum(scores))
        

                        
    ##
    # listAllRecurseMoves
    #   Description: FUNction finds all the valid movement moves
    #
    #   Param: current state
    #          All possible movement moves are made off of this
    #
    #   Return:
    #       Returns all possible movement moves

    def listAllRecurseMoves(self, currentState):
        result = listAllMovementMoves(currentState)
        result.append(Move(END, None, None))
        return result