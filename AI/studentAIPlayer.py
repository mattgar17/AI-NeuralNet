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
from random import seed
from random import random

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
        self.network = 
        [[ -8.50863524e-02   5.92700048e-01  -6.72884740e-01  -1.65707515e-01
           -6.54690046e-01  -7.50265839e-01  -1.96752868e-01  -2.83456515e-01
           -1.31496152e-01   7.06666688e-02]
         [ -7.30834672e-01   2.52494107e-01   1.91873857e-01   1.67578803e+00
           -1.14727183e+00   6.36978807e-01  -5.25988448e-01   4.69738220e-01
            1.44990530e-02  -1.44108196e+00]
         [  6.01489137e-01   9.36523151e-01  -3.73151644e-01   3.84645231e-01
            7.52778305e-01   7.89213327e-01  -8.29911577e-01  -9.21890434e-01
           -6.60339161e-01   7.56285007e-01]
         [ -7.85442005e-01  -1.52306000e-01   8.89851960e-01   7.13157032e-03
            3.95239726e-01  -3.74979031e-01   4.02459026e-01   6.61860084e-01
           -9.87969945e-01   5.33889405e-01]
         [  9.77722178e-01   4.96331309e-01  -4.39112016e-01   5.78558657e-01
           -7.93547987e-01  -1.04212948e-01   8.17191006e-01  -4.12771703e-01
           -4.24449323e-01  -7.39942856e-01]
         [ -9.61266084e-01   3.57671066e-01  -5.76743768e-01  -4.68906681e-01
           -1.68536814e-02  -8.93274910e-01   1.48235211e-01  -7.06542850e-01
            1.78611074e-01   3.99516720e-01]
         [ -7.95331142e-01  -1.71888024e-01   3.88800315e-01  -1.71641461e-01
           -9.00093082e-01   7.17928118e-02   3.27589290e-01   2.97782241e-02
            8.89189512e-01   1.73110081e-01]
         [  8.06803831e-01  -7.25050592e-01  -7.21447305e-01   6.14782577e-01
           -2.04646326e-01  -6.69291606e-01   8.55017161e-01  -3.04468281e-01
            5.01624206e-01   4.51995971e-01]
         [  7.66612182e-01   2.47344414e-01   5.01884868e-01  -3.02203316e-01
           -4.60144216e-01   7.91772436e-01  -1.43817620e-01   9.29680094e-01
            3.26882996e-01   2.43391440e-01]
         [ -7.70508054e-01   8.98978517e-01  -1.00175733e-01   1.56779229e-01
           -1.83726394e-01  -5.25946040e-01   8.06759041e-01   1.47358973e-01
           -9.94259346e-01   2.34289827e-01]
         [ -2.98466027e-01   6.94892909e-02   6.95328162e-01  -4.76865562e-01
            8.48971945e-01   2.17617537e-01  -9.23921262e-01   8.32124643e-01
            2.79313320e-01   1.15181492e+00]
         [ -6.55318983e-01  -7.25728501e-01   8.65190926e-01   3.93636323e-01
           -8.67999655e-01   5.10926105e-01   5.07752377e-01   8.46049071e-01
            4.23049517e-01  -7.51458076e-01]
         [ -3.76254891e-01  -6.98433907e-01  -1.29698673e+00  -9.46899597e-01
            9.30513944e-01  -1.18211698e-01   8.23075297e-01   3.91266337e-01
           -1.28338263e+00   1.97915498e-01]
         [  7.55503385e-01   1.18833562e+00  -2.31539271e-01  -1.40202715e+00
            8.11723392e-01  -7.29925279e-01   1.33164173e+00  -5.17074159e-01
            1.95354447e-01   1.13379143e+00]
         [  1.12480468e-01  -7.27089549e-01  -8.80164621e-01  -7.57313089e-01
           -9.10896243e-01  -7.85011742e-01  -5.48581323e-01   4.25977961e-01
            1.19433964e-01  -9.74888040e-01]
         [ -8.56051441e-01   9.34552660e-01   1.36200924e-01  -5.93413531e-01
           -4.95348511e-01   4.87651708e-01  -6.09141038e-01   1.62717855e-01
            9.40039978e-01   6.93657603e-01]
         [ -5.02440154e-01  -6.98182167e-03   2.13984336e-01   5.98762799e-01
           -6.74931712e-01  -9.68857889e-01  -8.30498542e-01  -3.47010381e-02
            1.88112424e-01   1.71303650e-01]
         [ -3.65275181e-01   9.77232309e-01   1.59490438e-01  -2.39717655e-01
            1.01896438e-01   4.90668862e-01   3.38465787e-01  -4.70160885e-01
           -8.67330331e-01  -2.59831604e-01]
         [  2.59435014e-01  -5.79651980e-01   5.05511107e-01  -8.66927037e-01
           -4.79369803e-01   6.09509127e-01  -6.13131435e-01   2.78921762e-01
            4.93406182e-02   8.49615941e-01]
         [ -4.73406459e-01  -8.68077819e-01   4.70131927e-01   5.44356059e-01
            8.15631705e-01   8.63944138e-01  -9.72096854e-01  -5.31275828e-01
            2.33556714e-01   8.98032641e-01]
         [  9.00352238e-01   1.13306376e-01   8.31212700e-01   2.83132418e-01
           -2.19984572e-01  -2.80186658e-02   2.08620966e-01   9.90958430e-02
            8.52362853e-01   8.37466871e-01]
         [ -1.62004605e-01   9.41898143e-01  -7.28644703e-01  -9.38746043e-01
           -6.97940041e-01  -1.77783637e-02  -9.12514137e-01   8.69190598e-01
            5.51750427e-01  -8.12792816e-01]
         [ -6.47607489e-01  -3.35872851e-01  -7.38006310e-01   6.18981384e-01
           -3.10526695e-01   8.80214965e-01   1.64028360e-01   7.57663969e-01
            6.89468891e-01   8.10784637e-01]
         [ -8.02394684e-02   9.26936320e-02   5.97207182e-01  -4.28562297e-01
           -1.94929548e-02   1.98220615e-01  -9.68933449e-01   1.86962816e-01
           -1.32647302e-01   6.14721058e-01]
         [ -9.38734094e-01   6.67832523e-01   9.38683788e-01   2.87573557e-01
            3.73811452e-01   5.20106141e-01  -1.25277951e+00   1.92745923e-01
            1.09186285e+00  -8.13815289e-05]
         [ -9.50951781e-01   9.68891384e-01  -3.23395407e-01   7.56161995e-01
            2.41333845e-01   6.28588921e-01   1.93859262e-01   2.29402572e-01
           -5.31327952e-01   3.30835904e-01]
         [  5.00043527e-01   7.16627673e-01   5.10164377e-01   3.96114497e-01
            7.28958860e-01  -3.54638006e-01   3.41577582e-01  -9.82521272e-02
           -2.35794496e-01  -1.78377300e-01]
         [  3.86944009e-01  -1.16087989e-01  -1.09760974e-01  -5.78827190e-01
            1.15806220e+00   1.59727956e-01   1.14571116e-01  -4.39393429e-01
           -8.45036782e-01   1.23482575e+00]
         [  7.77860905e-01   8.13162661e-01   2.99512524e-01  -5.18782476e-01
           -4.83781098e-01   7.03785592e-01   8.48864631e-02   5.96930908e-01
            1.20430535e-01   4.99885826e-01]
         [  3.80232549e-02   5.41767821e-01   1.37715981e-01  -6.85802428e-02
           -3.14622184e-01  -8.63581303e-01  -2.44151641e-01  -8.40747845e-01
            9.65634227e-01  -6.36774297e-01]
         [  5.44936950e-02   6.31978396e-01   1.15979586e+00   1.05854198e+00
           -8.80104142e-01   2.29803832e-01  -6.70253951e-01  -1.97561521e-01
            9.18748913e-01  -1.21274526e+00]]


        

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
        compositeScore = scores[1]/float(sum(scores))

        #convert currentState to input array
        inputArray = self.makeInputLayerArray(state)
        #Eval with NN and pass in compScore
        self.neuralEval(inputArray, compositeScore)
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




 
    # Initialize a network
    def initNetwork(numInputs, numLayers, numOutput):
        network = list()
        hidden_layer = [{'weights':[random() for i in range(numInputs + 1)]} for i in range(numLayers)]
        network.append(hidden_layer)
        output_layer = [{'weights':[random() for i in range(numLayers + 1)]} for i in range(numOutput)]
        network.append(output_layer)
        return network

    



    def makeInputLayerArray(self, currentState):

        inputArray = []
        playerFoodAmt = currentState.inventories[self.playerId].foodCount
        oppFoodAmt = currentState.inventories[(self.playerId + 1) % 2].foodCount

        #Bias for the arrays
        inputArray.append(1)

        #Sets the first 11 array values according to the amount of food player has
        for i in range(0, 11):
            if playerFoodAmt == i:
                inputArray.append(1)
            else:
                inputArray.append(0)

        for n in range(12, 23):
            if oppFoodAmt == (n-12):
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


        return inputArray


    # sigmoid function

    def nonlin(x,deriv=False):
        if(deriv==True):
            return x*(1-x)
        return 1/(1+np.exp(-x))

    #Make this into the function that takes the network and foreward propagates it
    #returning the eval score
    def neuralEval(self, inputArray, targetScore):
        
        # input dataset
        #result from State -> Array/List Function
        X = inputArray
 
        # seed random numbers to make calculation
        # deterministic (just a good practice)
        np.random.seed(1)
        seed(1)
        self.network = initNetwork(27, 10, 1)
        for layer in network:
            print(layer)
        # change to use currently defined weights
        # initialize weights randomly with mean 0
        weights1 = self.network.weights1
        weights2 = self.network.weights2
        print "WEIGHTS BEFORE"
        print weights1
        print weights2
        #weights1 = 2*np.random.random((31,10)) - 1
        #weights2 = 2*np.random.random((10,1)) - 1

        for j in xrange(1):
            # forward propagation
            l0 = X
            l1 = nonlin(np.dot(l0,weights1))
            l2 = nonlin(np.dot(l1,weights2))

            # how much did we miss? 
            l2_error = targetScore - l2
            if (j% 10000) == 0:
                print "Error:" + str(np.mean(np.abs(l2_error)))


            # in what direction is the target value?
            # were we really sure? if so, don't change too much.
            l2_delta = l2_error*nonlin(l2,deriv=True)

            # how much did each l1 value contribute to the l2 error (according to the weights)?

            l1_error = l2_delta.dot(weights2.T)
            # multiply how much we missed by the
            # slope of the sigmoid at the values in l1
            l1_delta = l1_error * nonlin(l1,True)
            # update weights    
            self.network.weights2 += l1.T.dot(l2_delta)
            self.network.weights1 += l0.T.dot(l1_delta)

        print "Output After Training:"
        print l2

        #REGISTER WIN
        #pass the adjusted weights from this game to the next

        #print out weights when we are done with games

        #Aim for error within 0.03 




    #Unit test/training
    #best and worst case
    X = np.array([[1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
                [1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
                [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
                [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1]])
    #change to target score from Heuristic eval
    # output dataset           
    y = np.array([[1,.5,.5,0]]).T

    # seed random numbers to make calculation
    # deterministic (just a good practice)
    np.random.seed(1)

    # initialize weights randomly with mean 0
    #weights1 = 2*np.random.random((31,10)) - 1
    #weights2 = 2*np.random.random((10,1)) - 1
    seed(1)
    network = initNetwork(27, 10, 1)
    for layer in network:
        print(layer)
    weights1 = network.hidden_layer[0]
    weights2 = network.hidden_layer[1]
    print "WEIGHTS BEFORE"
    print weights1
    print weights2
    for j in xrange(60000):
        # forward propagation
        l0 = X
        l1 = nonlin(np.dot(l0,weights1))
        l2 = nonlin(np.dot(l1,weights2))

        # how much did we miss? 
        l2_error = y - l2
        if (j% 10000) == 0:
            print "Error:" + str(np.mean(np.abs(l2_error)))


        # in what direction is the target value?
        # were we really sure? if so, don't change too much.
        l2_delta = l2_error*nonlin(l2,deriv=True)

        # how much did each l1 value contribute to the l2 error (according to the weights)?

        l1_error = l2_delta.dot(weights2.T)
        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * nonlin(l1,True)
        # update weights    
        weights2 += l1.T.dot(l2_delta)
        weights1 += l0.T.dot(l1_delta)

    print "Output After Training:"
    # print l0
    # print l1
    # print l2
    print weights1
    print "*****************************************************"
    print weights2