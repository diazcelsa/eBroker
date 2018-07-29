import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series(data, x, y, label=None, palette=None):
    # Visualize annotation
    plt.figure(figsize=(30, 10))
    if label:
        ax = sns.pointplot(x=x, y=y, data=data.loc[::10], hue=label, palette=palette)
    else:
        ax = sns.pointplot(x=x, y=y, data=data.loc[::10])
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % 10 == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.legend()
    _ = plt.xticks(rotation=90)
