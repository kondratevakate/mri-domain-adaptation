import matplotlib.pyplot as plt
from matplotlib.colors import rgb2hex

def get_colors(cmap, breaks):
    return [rgb2hex(cmap(bb)) for bb in breaks]

def display_brain(dataset, i):
    n = dataset[i][0]
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(n[n.shape[0] // 2, :, :])
    plt.subplot(132)
    plt.imshow(n[:, n.shape[1] // 2, :])
    plt.subplot(133)
    plt.imshow(n[:, :, n.shape[2] // 2])
    plt.show()

def display_brain_img(img, vmin=None, vmax=None, cmap="viridis", site="",
                     experiment = False, figure_name = ''):
    n = img[0]
    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    plt.imshow(n[n.shape[0] // 2, :, :].T[::-1], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.ylabel(site, fontdict={"size" : 20})
    plt.yticks([], [])
    plt.xticks([], [])
    plt.subplot(132)
    plt.imshow(n[:, n.shape[1] // 2, :].T[::-1], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(n[:, :, n.shape[2] // 2], vmin=vmin, vmax=vmax, cmap=cmap)
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    if experiment:
            experiment.log_figure(figure_name = figure_name)
    else:
        plt.show()
    
