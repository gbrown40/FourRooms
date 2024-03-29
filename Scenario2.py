from FourRooms import FourRooms
import numpy as np
import sys

def chooseAction(state, epsilon, Q_vals, numPackages, stochastic):
    action = 0
    if np.random.uniform(0, 1) <= epsilon:
         #choose random action
        action = np.random.choice(4)
    else:
        x = state[0]
        y = state[1]
        #choose action with highest expected value
        action = np.argmax(Q_vals[x][y][numPackages - 1][:])
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
    return action, epsilon

def getReward(numPackages, packagesRemaining, oldPos, newPos):
    #if package has been collected, reward
    #higher reward for each subsequent package found
    if (packagesRemaining == numPackages - 1):
        return 100 * (4 - numPackages)
    #if ran into a wall, punish
    elif (oldPos == newPos):
        return -50
    #small punishment for not finding package
    else: 
        return -1

def updateQ(state, new_state, action, reward, Q_vals, lr, gamma, numPackages, packagesRemaining, stochastic):
    x = state[0]
    y = state[1]
    xNew = new_state[0]
    yNew = new_state[1]
    #if package was just picked up, use q table for next package to find the expected reward of the next state
    state_package_pair = Q_vals[xNew][yNew][numPackages - 1]
    if packagesRemaining == numPackages - 1 and numPackages >= 0:
        Q_vals[xNew][yNew][numPackages - 2]
    #compute best action for next state
    future_exp_val = state_package_pair[:]
    future_action = np.argmax(future_exp_val)
    #stochastically choose the action for next state
    if stochastic:
        if future_action == 0:
            future_action = np.random.choice([0, 1, 2, 3], p=[0.8, 0.2/3.0, 0.2/3.0, 0.2/3.0])
        elif future_action == 1:
            future_action = np.random.choice([0, 1, 2, 3], p=[0.2/3.0, 0.8, 0.2/3.0, 0.2/3.0])
        elif future_action == 2:
            future_action = np.random.choice([0, 1, 2, 3], p=[0.2/3.0, 0.2/3.0, 0.8, 0.2/3.0])
        elif future_action == 3:
            future_action = np.random.choice([0, 1, 2, 3], p=[0.2/3.0, 0.2/3.0, 0.2/3.0, 0.8])
    #update Q values based on reward from taking action and expected reward of cell moved to from that action
    Q_vals[x][y][numPackages - 1][action] = Q_vals[x][y][numPackages - 1][action] + lr * (reward + gamma * state_package_pair[future_action] - Q_vals[x][y][numPackages - 1][action])
    return Q_vals

def main():
    #parameters
    aTypes = ['UP', 'DOWN', 'LEFT', 'RIGHT']
    gTypes = ['EMPTY', 'RED', 'GREEN', 'BLUE']
    lr = 0.5
    gamma = 0.9
    epsilon = 0.6
    #Q table that stores q values of state/package number/action pairs
    #each state/action pair has a different value for each number of packages left to collect so the algorithm does not revisit packages
    Q_vals = np.zeros((12, 12, 3, 4))
    num_episodes = 100
    #stochastic set to false unless stochastic flag specified
    stochastic = False
    if len(sys.argv) > 1:
        stochastic = True
    fourRoomsObj = FourRooms('multi')
    #training
    for i in range(0, num_episodes):
        #track number of packages to left collect
        numPackages = 3
        while not fourRoomsObj.isTerminal():
            #get current position
            state = fourRoomsObj.getPosition()
            #choose then take an action
            action, epsilon = chooseAction(state, epsilon, Q_vals, numPackages, stochastic)
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            #calculate reward
            reward = getReward(numPackages, packagesRemaining, state, newPos)
            #update Q values
            Q_vals = updateQ(state, newPos, action, reward, Q_vals, lr, gamma, numPackages, packagesRemaining, stochastic)
            #update number of packages left to collect
            if packagesRemaining == numPackages - 1:
                numPackages = numPackages - 1
            #decay epsilon
            epsilon = epsilon * 0.95
        print("epoch " + str(i))
        #start new epoch
        fourRoomsObj.newEpoch()

    #track number of packages left to collect
    numPackages = 3
    #run simulation
    while not fourRoomsObj.isTerminal():
        state = fourRoomsObj.getPosition()
        x = state[0]
        y = state[1]
        #chose optimal action based on learned Q table
        action = np.argmax(Q_vals[x][y][numPackages - 1][:])
        gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
        #update number of packages left to collect
        if packagesRemaining == numPackages - 1:
                numPackages = numPackages - 1
        print("Agent took {0} action and moved to {1} of type {2}".format (aTypes[action], newPos, gTypes[gridType]))

    fourRoomsObj.showPath(-1)

if __name__ == "__main__":
    main()