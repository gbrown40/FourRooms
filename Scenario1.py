from FourRooms import FourRooms
import numpy as np
import sys

def chooseAction(state, epsilon, Q_vals, stochastic):
    action = 0
    if np.random.uniform(0, 1) <= epsilon:
        #choose random action
        action = np.random.choice(4)
    else:
        x = state[0]
        y = state[1]
        #choose action with highest expected value
        action = np.argmax(Q_vals[x][y][:])
        #if stochastic, action with highest expected value has 80% probability of being chosen
        if stochastic:
            if action == 0:
                action = np.random.choice([0, 1, 2, 3], p=[0.8, 0.2/3.0, 0.2/3.0, 0.2/3.0])
            elif action == 1:
                action = np.random.choice([0, 1, 2, 3], p=[0.2/3.0, 0.8, 0.2/3.0, 0.2/3.0])
            elif action == 2:
                action = np.random.choice([0, 1, 2, 3], p=[0.2/3.0, 0.2/3.0, 0.8, 0.2/3.0])
            elif action == 3:
                action = np.random.choice([0, 1, 2, 3], p=[0.2/3.0, 0.2/3.0, 0.2/3.0, 0.8])
    return action

def getReward(packagesRemaining, oldPos,newPos):
    #if package has been collected, reward
    if (packagesRemaining == 0): # 
        return 100
    #if ran into a wall, punish
    elif (oldPos == newPos):
        return -50
    #small punishment for not finding package
    else: 
        return -1

def updateQ(state, new_state, action, reward, Q_vals, lr, gamma):
    x = state[0]
    y = state[1]
    xNew = new_state[0]
    yNew = new_state[1]
    #update Q values based on reward from taking action and expected reward of cell moved to from that action
    Q_vals[x][y][action] = Q_vals[x][y][action] + lr * (reward + gamma * np.max(Q_vals[xNew][yNew][:]) - Q_vals[x][y][action])
    return Q_vals

def main():
    #parameters
    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    lr = 0.2
    gamma = 0.8
    epsilon = 0.6
    #Q table that stores q values of state/action pairs
    Q_vals = np.zeros((12, 12, 4))
    num_episodes = 100
    #stochastic set to false unless stochastic flag specified
    stochastic = False
    if len(sys.argv) > 1:
        stochastic = True
    fourRoomsObj = FourRooms('simple')
    #training
    for i in range(0, num_episodes):
        while not fourRoomsObj.isTerminal():
            #get current position
            state = fourRoomsObj.getPosition()
            #choose then take an action
            action = chooseAction(state, epsilon, Q_vals, stochastic)
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            #calculate reward
            reward = getReward(packagesRemaining, state, newPos)
            #update Q values
            Q_vals = updateQ(state, newPos, action, reward, Q_vals, lr, gamma)
        print("epoch " + str(i))
        #start new epoch
        fourRoomsObj.newEpoch()

    #run simulation
    while not fourRoomsObj.isTerminal():
        state = fourRoomsObj.getPosition()
        x = state[0]
        y = state[1]
        #chose optimal action based on learned Q table
        action = np.argmax(Q_vals[x][y][:])
        gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
        print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[action], newPos, gTypes[gridType]))

    fourRoomsObj.showPath(-1)

if __name__ == "__main__":
    main()