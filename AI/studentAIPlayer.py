
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
from random import *

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
        super(AIPlayer,self).__init__(inputPlayerId, "MiniMattNeuralNet")
        #limit of how deep function should recurse
        self.depthLimit = 3
        self.me = inputPlayerId
        self.currGame = 0
        self.trainGames = 2
        self.layer1 = None
        self.layer2 = None
        #hardcoded weights from series of training
        self.weights1 = np.matrix( \
            [[ -2.06603621e+00,   1.30985206e+00,  -1.08793555e+01, 
          -1.01429420e+01,  -1.06288020e+01,  -1.08488467e+01, 
          -9.35555544e+00,  -1.04687644e+01,  -1.02836627e+01, 
           5.56545678e-02], 
        [ -7.30834672e-01,   2.52494107e-01,   1.91873857e-01, 
           1.67578803e+00,  -1.14727183e+00,   6.36978807e-01, 
          -5.25988448e-01,   4.69738220e-01,   1.44990530e-02, 
          -1.44108196e+00], 
        [ -8.96313751e+00,  -5.95797478e+00,  -9.67322531e+00, 
          -8.65986197e+00,  -8.60747566e+00,  -8.45447189e+00, 
          -1.02190194e+01,  -1.02256503e+01,  -9.73947040e+00, 
          -3.49707994e+00], 
        [ -2.16249306e+01,  -1.93433492e+00,  -2.01680300e+01, 
          -2.11004005e+01,  -2.05718717e+01,  -2.14842734e+01, 
          -2.02708241e+01,  -2.04189807e+01,  -2.20434899e+01, 
          -6.25112426e-01], 
        [  1.89628589e+00,   3.22639748e+00,  -1.52241896e+01, 
          -1.41995954e+01,  -1.55872170e+01,  -1.49002169e+01, 
          -1.35888467e+01,  -1.52057780e+01,  -1.51775053e+01, 
          -2.31024703e+00], 
        [  1.93917414e+00,   7.05825300e-01,   1.87869473e+00, 
           1.98924611e+00,   2.44328362e+00,   1.57161577e+00, 
           2.55131824e+00,   1.75530385e+00,   2.67180223e+00, 
           1.52788330e-01], 
        [  4.55844422e+00,   4.94725975e-01,   2.93978585e+00, 
           2.38081501e+00,   1.66641851e+00,   2.62793361e+00, 
           2.99276355e+00,   2.58941759e+00,   3.30449988e+00, 
           3.68419743e-02], 
        [  3.98103254e+00,   5.07028605e-01,   2.33106862e+00, 
           3.67071692e+00,   2.86619491e+00,   2.37702311e+00, 
           4.17902898e+00,   2.74501625e+00,   3.61567440e+00, 
           9.08916262e-01], 
        [  3.38888865e+00,   7.33734230e-01,   5.76039032e+00, 
           4.95707747e+00,   4.81340327e+00,   6.05822394e+00, 
           5.09839869e+00,   6.18515610e+00,   5.56221462e+00, 
           1.20621768e+00], 
        [  2.75323400e+00,   1.11257630e+00,   5.62467332e+00, 
           5.85480228e+00,   5.54828835e+00,   5.20728416e+00, 
           6.50304899e+00,   5.86630362e+00,   4.72478620e+00, 
           1.59969414e+00], 
        [  3.31868671e+00,   9.00886116e-01,   6.69016176e+00, 
           5.53282270e+00,   6.86113338e+00,   6.24016267e+00, 
           5.06705070e+00,   6.83107881e+00,   6.23383396e+00, 
           1.77804938e+00], 
        [  3.24005034e+00,   1.06162856e+00,   6.32303789e+00, 
           5.87304279e+00,   4.72577804e+00,   6.02976256e+00, 
           6.10177127e+00,   6.35584931e+00,   5.78786750e+00, 
           1.68276054e+00], 
        [  2.01793067e+00,   3.79513255e-01,   3.07276717e+00, 
           3.42128329e+00,   5.29662505e+00,   4.25194710e+00, 
           5.14538497e+00,   4.75758565e+00,   3.08399726e+00, 
           1.69585884e+00], 
        [  4.25041127e-01,   1.14527952e+00,  -4.81529272e+00, 
          -5.98174508e+00,  -3.66804578e+00,  -5.35920415e+00, 
          -2.26504815e+00,  -5.11544146e+00,  -4.31569871e+00, 
           5.94797354e+00], 
        [  8.77616490e-01,   8.32204245e-02,  -3.70024201e+00, 
          -3.39050027e+00,  -3.75712661e+00,  -3.53142625e+00, 
          -3.46025582e+00,  -2.44517026e+00,  -2.62009111e+00, 
          -3.71318168e+00], 
        [ -3.27167507e+00,   8.84450787e-01,  -2.66643901e+00, 
          -3.35774288e+00,  -3.14346091e+00,  -2.23523574e+00, 
          -3.25957924e+00,  -2.55307447e+00,  -1.96154833e+00, 
          -1.39724298e+00], 
        [ -5.02440154e-01,  -6.98182167e-03,   2.13984336e-01, 
           5.98762799e-01,  -6.74931712e-01,  -9.68857889e-01, 
          -8.30498542e-01,  -3.47010381e-02,   1.88112424e-01, 
           1.71303650e-01], 
        [ -3.65275181e-01,   9.77232309e-01,   1.59490438e-01, 
          -2.39717655e-01,   1.01896438e-01,   4.90668862e-01, 
           3.38465787e-01,  -4.70160885e-01,  -8.67330331e-01, 
          -2.59831604e-01], 
        [  2.59435014e-01,  -5.79651980e-01,   5.05511107e-01, 
          -8.66927037e-01,  -4.79369803e-01,   6.09509127e-01, 
          -6.13131435e-01,   2.78921762e-01,   4.93406182e-02, 
           8.49615941e-01], 
        [ -4.73406459e-01,  -8.68077819e-01,   4.70131927e-01, 
           5.44356059e-01,   8.15631705e-01,   8.63944138e-01, 
          -9.72096854e-01,  -5.31275828e-01,   2.33556714e-01, 
           8.98032641e-01], 
        [  9.00352238e-01,   1.13306376e-01,   8.31212700e-01, 
           2.83132418e-01,  -2.19984572e-01,  -2.80186658e-02, 
           2.08620966e-01,   9.90958430e-02,   8.52362853e-01, 
           8.37466871e-01], 
        [ -1.62004605e-01,   9.41898143e-01,  -7.28644703e-01, 
          -9.38746043e-01,  -6.97940041e-01,  -1.77783637e-02, 
          -9.12514137e-01,   8.69190598e-01,   5.51750427e-01, 
          -8.12792816e-01], 
        [ -6.47607489e-01,  -3.35872851e-01,  -7.38006310e-01, 
           6.18981384e-01,  -3.10526695e-01,   8.80214965e-01, 
           1.64028360e-01,   7.57663969e-01,   6.89468891e-01, 
           8.10784637e-01], 
        [ -8.02394684e-02,   9.26936320e-02,   5.97207182e-01, 
          -4.28562297e-01,  -1.94929548e-02,   1.98220615e-01, 
          -9.68933449e-01,   1.86962816e-01,  -1.32647302e-01, 
           6.14721058e-01], 
        [ -9.38734094e-01,   6.67832523e-01,   9.38683788e-01, 
           2.87573557e-01,   3.73811452e-01,   5.20106141e-01, 
          -1.25277951e+00,   1.92745923e-01,   1.09186285e+00, 
          -8.13815289e-05], 
        [ -1.66601572e+00,   2.44899905e+00,  -2.92623044e+00, 
          -1.84306918e+00,  -2.33094415e+00,  -1.97093024e+00, 
          -2.53956581e+00,  -2.38610885e+00,  -3.12178159e+00, 
          -5.95195928e-01], 
        [  4.28226159e+00,   2.54534246e+00,  -5.82510028e+00, 
          -5.94319602e+00,  -5.60829747e+00,  -6.70756248e+00, 
          -5.27521185e+00,  -6.44063401e+00,  -6.56261080e+00, 
          -1.28078757e+00], 
        [  2.56454995e+00,   1.33042906e+00,  -1.23098371e+00, 
          -1.70330118e+00,   7.78096628e-02,  -9.64620874e-01, 
          -1.02293677e+00,  -1.56155213e+00,  -1.96887254e+00, 
           1.77463215e-01], 
        [ -9.73779319e-01,  -4.08455475e-01,  -2.62497529e+00, 
          -3.43950722e+00,  -3.40991201e+00,  -2.20676330e+00, 
          -2.87342801e+00,  -2.30844430e+00,  -2.78809554e+00, 
           5.88928947e-01], 
        [ -8.57204556e-01,  -9.54522181e-01,   2.23269968e+00, 
           2.01500810e+00,   1.78209136e+00,   1.17863587e+00, 
           1.96848640e+00,   1.22043691e+00,   3.13128572e+00, 
           1.22336994e+00], 
        [ -2.90909214e+00,  -3.10076542e-01,   6.54176701e+00, 
           6.68378045e+00,   4.65787368e+00,   5.77913447e+00, 
           4.92730474e+00,   5.23954433e+00,   6.23410466e+00, 
           6.60117123e-01]])

   

        self.weights2= np.matrix(\
           [[ 0.49991632], 
            [ 0.97663068], 
            [-1.05506719], 
            [-2.438824  ], 
            [ 1.91679572], 
            [-0.50041813], 
            [ 2.33412594], 
            [ 0.38283727], 
            [ 0.04740067], 
            [ 1.13564406]])

        

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
         #keep track of these
        self.myFood = None
        self.myFood1 = None
        self.myFood2 = None
        self.myTunnel = None
        self.enemyAnthill = None
        self.enemyTunnel = None
        self.constrCoords = None

        if (currentState.whoseTurn == PLAYER_TWO):
            #we are player 1
            enemy = 1
        else:
            enemy = 0

        if currentState.phase == SETUP_PHASE_1:
            return [(2,1), (7, 2),
                    (6,3), (5,3), (0,3), (1,3), \
                    (2,3), (3,3), (4,3), \
                    (5,0), (9,0) ];

        #set the enemy food to the least optimal places
        elif currentState.phase == SETUP_PHASE_2:
            self.enemyAnthill = getConstrList(currentState, enemy, (ANTHILL,))[0]
            self.enemyTunnel = getConstrList(currentState, enemy, (TUNNEL,))[0]
            numToPlace = 2
            moves = []
            for i in range(0, numToPlace):
                for j in range(0,9):
                    for k in range (6,9):
                        move = None
                        #Find the farthest open space
                        if(((stepsToReach(currentState, (j,k), self.enemyTunnel.coords))\
                            + (stepsToReach(currentState, (j,k), self.enemyAnthill.coords)))
                            > ((stepsToReach(currentState, moves, self.enemyTunnel.coords))\
                            + (stepsToReach(currentState, moves, self.enemyAnthill.coords)))):
                            #Set the move if this space is empty
                            if currentState.board[j][k].constr == None and (j, k) not in moves:
                                move = (j, k)
                                #Just need to make the space non-empty. So I threw whatever I felt like in there.
                                currentState.board[j][k].constr == True
                                moves.append(move)
            return moves
    
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
    #     #Attack a random enemy.
        return enemyLocations[0]  #don't care
    #     return enemyLocations[np.random.randint(0, len(enemyLocations) - 1)]

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
                            #change the beta value if the new best value is less than the current minimum value
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
    #Heuristic evaluation 
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
        compositeScore = scores[1]/float(sum(scores))
        #compositeScore = 0
    #Neural Evaluation
        #convert currentState to input array for Neural Evaluation
        inputArray = self.makeInputLayerArray(state)
        #Eval with NN and return that score instead
        neuralScore = self.neuralEval(inputArray, compositeScore)
        #return the evaluated score from the network
        return neuralScore
                        
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


    #funciton that takes a gamestate and returns our array representation of that state
    def makeInputLayerArray(self, currentState):

        inputArray = []
        playerFoodAmt = currentState.inventories[self.playerId].foodCount
        oppFoodAmt = currentState.inventories[(self.playerId + 1) % 2].foodCount

        #Bias for the arrays
        inputArray.append(1)

        #Sets the first 11 array values according to the amount of food player has
        for i in range(0, 12):
            if playerFoodAmt == i:
                inputArray.append(1)
            else:
                inputArray.append(0)

        for n in range(0, 12):
            if oppFoodAmt == (n):
                inputArray.append(1)
            else:
                inputArray.append(0)

        for a in currentState.inventories[self.playerId].ants:

            if a.type == WORKER:
                #If the ant is carrying food, append 1 to how close they are to the food, or 0 if it's not within that rangee
                if a.carrying:
                    if approxDist(a.coords, currentState.inventories[self.playerId].getAnthill().coords) > 5:
                        inputArray.append(1)
                    else:
                        inputArray.append(0)
                    if approxDist(a.coords, currentState.inventories[self.playerId].getAnthill().coords) <= 5 and approxDist(a.coords, currentState.inventories[self.playerId].getAnthill().coords)> 2:
                        inputArray.append(1)
                    else:
                        inputArray.append(0)
                    if approxDist(a.coords, currentState.inventories[self.playerId].getAnthill().coords) <= 2 and approxDist(a.coords, currentState.inventories[self.playerId].getAnthill().coords)> 0:
                        inputArray.append(1)
                    else:
                        inputArray.append(0)
                #if the worker ants aren't carrying anything, append 3 0's
                else:
                    inputArray.append(0)
                    inputArray.append(0)
                    inputArray.append(0)

                if not a.carrying:
                    if approxDist(a.coords, getConstrList(currentState, None, (FOOD,))[0].coords) > 5:
                        inputArray.append(1)
                    else:
                        inputArray.append(0)
                    if approxDist(a.coords, getConstrList(currentState, None, (FOOD,))[0].coords) <= 5 and approxDist(a.coords, getConstrList(currentState, None, (FOOD,))[0].coords)> 2:
                        inputArray.append(1)
                    else:
                        inputArray.append(0)
                    if approxDist(a.coords, getConstrList(currentState, None, (FOOD,))[0].coords) <= 2 and approxDist(a.coords, getConstrList(currentState, None, (FOOD,))[0].coords) > 0:
                        inputArray.append(1)
                    else:
                        inputArray.append(0)
                        # if the worker ants aren't carrying anything, append 3 0's
                else:
                    inputArray.append(0)
                    inputArray.append(0)
                    inputArray.append(0)

        if len(inputArray) != 31:
            print len(inputArray)
        while (len(inputArray) != 31):
            inputArray.append(0)
        return inputArray


    # sigmoid function
    def nonlin(self,x,deriv=False):
        if(deriv==True):
            return (x*(1-x))
        return 1/(1+np.exp(-x))

    #Make this into the function that takes the network and foreward propagates it
    #returning the eval score
    def neuralEval(self, inputArray, targetScore):
        for j in xrange(1):
            # forward propagation
            l0 = np.matrix(inputArray)
            if self.layer1 == None:
                l1 = self.nonlin(np.dot(l0,self.weights1))
                l2 = self.nonlin(np.dot(l1,self.weights2))
            else:
                #use updated weights for next game
                print "NEWGAME"
                l1 = self.nonlin(np.dot(l0,self.layer1))
                l2 = self.nonlin(np.dot(l1,self.layer2))

            # Calculate output error
            l2_error = targetScore - l2
            if (j% 10000) == 0:
                print "Error:" + str(np.mean(np.abs(l2_error)))
                print repr(l2)

            #calculate error and delta values for back propagation
            l2_delta = l2_error*(self.nonlin(l2,deriv=True))
            l1_error = l2_delta.dot(self.weights2.T)
            l1_delta = l1_error*(l1.T*(1-l1))

            # update weights for training 
            #self.weights2 += l1.getT().dot(l2_delta)
            #self.weights1 += l0.getT().dot(l1_delta)
        return l2[0,0]


    #REGISTER WIN called when  game is over
    def registerWin(self, hasWon):
        #pass the adjusted weights from this game to the next
        self.layer1 = self.weights1
        self.layer2 = self.weights2
        self.currGame += 1
        #print out weights when we are done with games
       
        print repr(self.layer1)
        print repr(self.layer2)
        #Aim for error within 0.03 
        pass


    # #Unit test/training
    # Basis for neuralEval function 
    # Algorithm form:
    # http://iamtrask.github.io/2015/07/12/basic-python-network/
    # #best and worst case and a couple middle cases
    # X = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
    #             [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
    #             [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
    #             [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1]])
    # #change to target score from Heuristic eval
    # # output dataset           
    # y = np.array([[1,.5,.5,0]]).T

    # # seed random numbers to make calculation
    # # deterministic (just a good practice)
    # np.random.seed(1)

    # # initialize weights randomly with mean 0
    # weights1 = 2*np.random.random((31,10)) - 1
    # weights2 = 2*np.random.random((10,1)) - 1
    # seed(1)
    # network = initNetwork(27, 10, 1)
    # for layer in network:
    #     print(layer)
    # #weights1 = network.weights1
    # #weights2 = network.weights2
    # print "WEIGHTS BEFORE"
    # print weights1
    # print weights2
    # for j in xrange(60000):
    #     # forward propagation
    #     l0 = X
    #     l1 = nonlin(np.dot(l0,weights1))
    #     l2 = nonlin(np.dot(l1,weights2))

    #     # how much did we miss? 
    #     l2_error = y - l2
    #     if (j% 10000) == 0:
    #         print "Error:" + str(np.mean(np.abs(l2_error)))


    #     # in what direction is the target value?
    #     # were we really sure? if so, don't change too much.
    #     l2_delta = l2_error*nonlin(l2,deriv=True)

    #     # how much did each l1 value contribute to the l2 error (according to the weights)?

    #     l1_error = l2_delta.dot(weights2.T)
    #     # multiply how much we missed by the
    #     # slope of the sigmoid at the values in l1
    #     l1_delta = l1_error * nonlin(l1,True)
    #     # update weights    
    #     weights2 += l1.T.dot(l2_delta)
    #     weights1 += l0.T.dot(l1_delta)

    # print "Output After Training:"
    # # print l0
    # # print l1
    # # print l2
    # print weights1
    # print "*****************************************************"
    # print weights2