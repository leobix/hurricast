import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


def animate_ouragan(vision_data, n=0):
    fig = plt.figure()
    ims = []
    for i in range(vision_data.shape[1]):
        #Show the geopotential--> 0
        #Show altitude -->1
        data = vision_data[n, i, 0, 1]
        img = plt.imshow(data, animated=True)
        ims.append([img])

    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                    repeat_delay=1000)
    return ani
