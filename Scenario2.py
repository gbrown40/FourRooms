from FourRooms import FourRooms
import numpy as np

def chooseAction(state, epsilon, Q_vals):
        # choose action with most expected value
        action = 0
        if np.random.uniform(0, 1) <= epsilon:
            action = np.random.choice(4)
        else:
            x = state[0]
            y = state[1]
            action = np.argmax(Q_vals[x][y][:])
        return action

def getReward(numPackages, packagesRemaining, oldPos,newPos, seen):
        '''reward function for finding packages'''
        # if running into a wall punish 
        # if package then reward 
        if newPos in seen:
            return -50, numPackages
        elif (packagesRemaining == numPackages - 1): # 
            if (packagesRemaining == 2):
                return 100, numPackages - 1
            elif (packagesRemaining == 1):
                return 150, numPackages - 1
            else:
                return 200, numPackages - 1
        elif (oldPos == newPos): # ran into a wall and did not move 
            return -50, numPackages
        else: 
            return 0, numPackages

def updateQ(state, new_state, action, reward, Q_vals, lr, gamma):
    x = state[0]
    y = state[1]
    xNew = new_state[0]
    yNew = new_state[1]
    Q_vals[x][y][action] = Q_vals[x][y][action] + lr * (reward + gamma * np.max(Q_vals[xNew][yNew][:]) - Q_vals[x][y][action])
    return Q_vals

def main():
    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    lr = 0.2
    gamma = 0.8
    epsilon = 0.6
    Q_vals = np.zeros((12, 12, 4))
    #train
    num_episodes = 100
    fourRoomsObj = FourRooms('simple')
    for i in range(0, num_episodes):
        numPackages = 3
        seen = []
        while not fourRoomsObj.isTerminal():
            state = fourRoomsObj.getPosition()
            action = chooseAction(state, epsilon, Q_vals)
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            if packagesRemaining == numPackages - 1:
                seen.append(newPos)
            reward, numPackages = getReward(numPackages, packagesRemaining, state, newPos, seen)
            Q_vals = updateQ(state, newPos, action, reward, Q_vals, lr, gamma)
        print("epoch " + str(i))
        fourRoomsObj.newEpoch()

    while not fourRoomsObj.isTerminal():
        state = fourRoomsObj.getPosition()
        x = state[0]
        y = state[1]
        action = np.argmax(Q_vals[x][y][:])
        gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
        print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[action], newPos, gTypes[gridType]))

    fourRoomsObj.showPath(-1)

if __name__ == "__main__":
    main()