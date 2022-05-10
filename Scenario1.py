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
        while not fourRoomsObj.isTerminal():
            state = fourRoomsObj.getPosition()
            action = chooseAction(state, epsilon, Q_vals)
            gridType, newPos, packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)

if __name__ == "__main__":
    main()