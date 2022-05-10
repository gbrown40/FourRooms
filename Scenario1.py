from FourRooms import FourRooms
import numpy as np

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

if __name__ == "__main__":
    main()