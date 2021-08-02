from train import Trainer
import matplotlib.pyplot as plt
import numpy as np


def main():
    trainer = Trainer("CartPole-v0")
    scores = trainer.reinforce()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()
    trainer.export_video()


if __name__ == "__main__":
    main()
