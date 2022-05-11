from FourRooms import FourRooms
import numpy as np

def chooseAction(state, epsilon, Q_vals, numPackages):
        # choose action with most expected value
        action = 0
        if np.random.uniform(0, 1) <= epsilon:
            action = np.random.choice(4)
            epsilon = epsilon - 0.001
        else:
            x = state[0]
            y = state[1]
            action = np.argmax(Q_vals[x][y][numPackages - 1][:])
        return action, epsilon

def getReward(numPackages, packagesRemaining, oldPos, newPos):
        '''reward function for finding packages'''
        # if running into a wall punish 
        # if package then reward 
        if (packagesRemaining == numPackages - 1): # 
            return 100
        elif (oldPos == newPos): # ran into a wall and did not move 
            return -50
        else: 
            return 0

def updateQ(state, new_state, action, reward, Q_vals, lr, gamma, numPackages):
    x = state[0]
    y = state[1]
    xNew = new_state[0]
    yNew = new_state[1]
    Q_vals[x][y][numPackages - 1][action] = Q_vals[x][y][numPackages - 1][action] + lr * (reward + gamma * np.max(Q_vals[xNew][yNew][numPackages - 1][:]) - Q_vals[x][y][numPackages - 1][action])
    return Q_vals

def main():
    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    lr = 0.2
    gamma = 0.8
    epsilon = 0.6
    Q_vals = np.zeros((12, 12, 3, 4))
    #train
    num_episodes = 1000
    fourRoomsObj = FourRooms('multi')
    for i in range(0, num_episodes):
        numPackages = 3
        while not fourRoomsObj.isTerminal():
            state = fourRoomsObj.getPosition()
            action, epsilon = chooseAction(state, epsilon, Q_vals, numPackages)
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            reward = getReward(numPackages, packagesRemaining, state, newPos)
            Q_vals = updateQ(state, newPos, action, reward, Q_vals, lr, gamma, numPackages)
            if packagesRemaining == numPackages - 1:
                numPackages = numPackages - 1
        print("epoch " + str(i))
        fourRoomsObj.newEpoch()

    numPackages = 3
    while not fourRoomsObj.isTerminal():
        state = fourRoomsObj.getPosition()
        x = state[0]
        y = state[1]
        action = np.argmax(Q_vals[x][y][numPackages - 1][:])
        gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
        if packagesRemaining == numPackages - 1:
                numPackages = numPackages - 1
        #fourRoomsObj.showPath(-1)
        print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[action], newPos, gTypes[gridType]))

    fourRoomsObj.showPath(-1)

if __name__ == "__main__":
    main()