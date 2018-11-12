import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter



def plot_keplen_meier_curves(clusters, status, survival, ax, filename=""):
    kmf = KaplanMeierFitter()
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10))
    for i in np.arange(0, 11):
        group =  np.where(cv.labels_ == i)[0]
        if len(group) == 0:
            continue
        how_many = len(group)
        death = np.where(status==1)[0].shape[0]
        kmf.fit(status[group, 0], y[stage_4][group, 1],
                label='Clust'+str(i)+": "+str(how_many)+"("+str(death)+")")
        ax = kmf.plot(ax=ax, ci_show=False)
    if filename != "":
        plt.savefig(filename)
    plt.show()


def plot_jaccard_matrix(M, labels_x, labels_y, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(15,10))

    ax.imshow(M, vmin=0, vmax=1)
    ax.set_xticks(labels_x)
    ax.set_yticks(labels_y)
    ax.colorbar()
    plt.show()
