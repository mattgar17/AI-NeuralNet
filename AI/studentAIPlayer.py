
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
            [[ -4.77386741e-01,   2.29288280e+00,  -9.92966074e+00,
                      -9.17126974e+00,  -9.70811475e+00,  -9.92530154e+00,
                      -8.38703673e+00,  -9.49442783e+00,  -9.28327475e+00,
                       8.84240655e-01],
                    [ -7.30834672e-01,   2.52494107e-01,   1.91873857e-01,
                       1.67578803e+00,  -1.14727183e+00,   6.36978807e-01,
                      -5.25988448e-01,   4.69738220e-01,   1.44990530e-02,
                      -1.44108196e+00],
                    [ -7.85817106e+00,  -5.17749038e+00,  -8.56297081e+00,
                      -7.54960747e+00,  -7.49722116e+00,  -7.34421739e+00,
                      -9.10876487e+00,  -9.11539581e+00,  -8.62921590e+00,
                      -3.51233175e+00],
                    [ -1.84607715e+01,  -1.45121245e+00,  -1.70038708e+01,
                      -1.79362413e+01,  -1.74077125e+01,  -1.83201142e+01,
                      -1.71066649e+01,  -1.72548215e+01,  -1.88793307e+01,
                      -1.87339229e-01],
                    [  3.33835460e+00,   3.24570039e+00,  -1.12629830e+01,
                      -1.02383888e+01,  -1.16260104e+01,  -1.09390103e+01,
                      -9.62764013e+00,  -1.12445714e+01,  -1.12162987e+01,
                      -1.19923667e+00],
                    [  2.35675865e+00,   7.62311103e-01,   3.25408770e+00,
                       3.36424022e+00,   3.81875527e+00,   2.94726464e+00,
                       3.92208990e+00,   3.13078700e+00,   4.04497949e+00,
                       5.17163399e-01],
                    [  4.73883270e+00,   5.84405436e-01,   4.55448288e+00,
                       3.99538736e+00,   3.28220357e+00,   4.24351330e+00,
                       4.59953092e+00,   4.20459766e+00,   4.91458407e+00,
                       5.21183729e-01],
                    [  4.01923114e+00,   5.06707334e-01,   2.22590357e+00,
                       3.56515839e+00,   2.76121362e+00,   2.27196042e+00,
                       4.06954179e+00,   2.64005759e+00,   3.50825375e+00,
                       1.13501563e+00],
                    [  2.95472950e+00,   6.75946013e-01,   4.67762535e+00,
                       3.87750511e+00,   3.72315934e+00,   4.97285621e+00,
                       4.01901080e+00,   5.11057414e+00,   4.49599435e+00,
                       1.12853173e+00],
                    [  1.71811025e+00,   1.05301914e+00,   3.60063188e+00,
                       3.84605642e+00,   3.52403745e+00,   3.18356527e+00,
                       4.45569634e+00,   3.85048790e+00,   2.70470589e+00,
                       1.21167808e+00],
                    [  1.94705514e+00,   7.82318771e-01,   3.94784249e+00,
                       2.78370670e+00,   4.10744235e+00,   3.48154044e+00,
                       2.32902151e+00,   4.09029839e+00,   3.51342210e+00,
                       1.25003173e+00],
                    [  1.87093911e+00,   9.42760654e-01,   2.74338974e+00,
                       2.30337496e+00,   1.13277982e+00,   2.44166023e+00,
                       2.54012663e+00,   2.77973735e+00,   2.22443392e+00,
                       1.03518893e+00],
                    [  1.46923998e+00,   2.88570853e-01,   2.33069056e+00,
                       2.68042939e+00,   4.55660070e+00,   3.50951726e+00,
                       4.43664598e+00,   4.01788744e+00,   2.34307018e+00,
                       1.55738825e+00],
                    [  1.85207630e+00,   2.17116415e+00,  -4.93368766e+00,
                      -6.10990067e+00,  -3.78849847e+00,  -5.48639548e+00,
                      -2.36084412e+00,  -5.24027828e+00,  -4.42830456e+00,
                       6.00627443e+00],
                    [  5.70612309e-01,   1.13830442e-02,  -3.35714387e+00,
                      -3.01336983e+00,  -3.44554102e+00,  -3.20754650e+00,
                      -3.11958546e+00,  -2.07211339e+00,  -2.22673606e+00,
                      -3.23030470e+00],
                    [ -2.80305659e+00,   9.13434282e-01,  -1.94144744e+00,
                      -2.63504550e+00,  -2.41390656e+00,  -1.50837904e+00,
                      -2.53593492e+00,  -1.82695798e+00,  -1.24190959e+00,
                      -1.10983476e+00],
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
                    [ -1.83323272e+00,   2.41383782e+00,  -3.80277512e+00,
                      -2.71960838e+00,  -3.20747536e+00,  -2.84747334e+00,
                      -3.41591894e+00,  -3.26263824e+00,  -3.99832358e+00,
                      -3.89157337e-01],
                    [  4.50835553e+00,   2.46631329e+00,  -5.49066137e+00,
                      -5.60875714e+00,  -5.27385840e+00,  -6.37312364e+00,
                      -4.94076987e+00,  -6.10619499e+00,  -6.22817188e+00,
                      -1.25982019e+00],
                    [  3.13172029e+00,   1.46244207e+00,  -2.09863292e-01,
                      -6.82171609e-01,   1.09894236e+00,   5.64784464e-02,
                      -1.66129977e-03,  -5.40446458e-01,  -9.47766646e-01,
                       5.54683448e-01],
                    [ -1.13516551e+00,  -3.06374388e-01,  -3.01883925e+00,
                      -3.83337265e+00,  -3.80377580e+00,  -2.60062738e+00,
                      -3.26723470e+00,  -2.70229800e+00,  -3.18195752e+00,
                       5.82020904e-01],
                    [ -2.45464433e-01,  -4.37728086e-01,   2.71935439e+00,
                       2.50151995e+00,   2.26894286e+00,   1.66506010e+00,
                       2.46225402e+00,   1.70713893e+00,   3.61894986e+00,
                       1.39563564e+00],
                    [ -3.07637760e+00,  -7.83635664e-02,   6.26806039e+00,
                       6.43217500e+00,   4.35486442e+00,   5.47958005e+00,
                       4.66409357e+00,   4.99050180e+00,   6.01022370e+00,
                       2.36430582e-01]])

   

        self.weights2= np.matrix(\
           [[ 0.33614915],
            [ 1.18593395],
            [-1.10468482],
            [-2.52514263],
            [ 1.90644283],
            [-0.52167766],
            [ 2.27232407],
            [ 0.29849388],
            [-0.06612088],
            [ 1.51622961]])

        

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
        print "MAKIN MOVES"
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
        #Heuristic evaluation commented out
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

        #convert currentState to input array
        inputArray = self.makeInputLayerArray(state)
        #Eval with NN and return that score instead
        neuralScore = self.neuralEval(inputArray, compositeScore)
        return compositeScore

                        
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




 
    # # Initialize a network
    # def initNetwork(numInputs, numLayers, numOutput):
    #     network = list()
    #     hidden_layer = [{'weights':[random() for i in range(numInputs + 1)]} for i in range(numLayers)]
    #     network.append(hidden_layer)
    #     output_layer = [{'weights':[random() for i in range(numLayers + 1)]} for i in range(numOutput)]
    #     network.append(output_layer)
    #     return network

    



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
        #print weights1
        #print weights2
        #self.weights1 = 2*np.random.random((31,10)) - 1
        #self.weights2 = 2*np.random.random((10,1)) - 1

        #print len(inputArray)
        #print self.weights1.shape
        #print self.weights2.shape
        for j in xrange(1):
            # forward propagation
            l0 = np.matrix(inputArray)
            if self.layer1 == None:
                l1 = self.nonlin(np.dot(l0,self.weights1))
                l2 = self.nonlin(np.dot(l1,self.weights2))
            else:
                print "NEWGAME"
                l1 = self.nonlin(np.dot(l0,self.layer1))
                l2 = self.nonlin(np.dot(l1,self.layer2))
            # Error, removed without heuristic
            l2_error = targetScore - l2
            if (j% 10000) == 0:
                print "Error:" + str(np.mean(np.abs(l2_error)))


            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            l2_delta = l2_error*(self.nonlin(l2,deriv=True))
            #print l2.shape
            # how much did each l1 value contribute to the l2 error (according to the weights)?
            l1_error = l2_delta.dot(self.weights2.T)
            # multiply how much we missed by the
            # slope of the sigmoid at the values in l1
            #temp = (self.nonlin(l1,deriv=True))
            #print l1.shape
            temp = (l1.T*(1-l1))
            l1_delta = l1_error*temp

            # update weights    
            self.weights2 += l1.getT().dot(l2_delta)
            self.weights1 += l0.getT().dot(l1_delta)
        #print l2[0,0]
        return l2[0,0]
        #print "Output After Training:"
        #print l2

    #REGISTER WIN
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