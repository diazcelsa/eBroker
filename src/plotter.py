import matplotlib.pyplot as plt
import seaborn as sns


def plot_time_series(data, x, y, label=None, palette=None, period=10):
    # Visualize annotation
    plt.figure(figsize=(30, 10))
    if label:
        ax = sns.pointplot(x=x, y=y, data=data.loc[::period], hue=label, palette=palette)
    else:
        ax = sns.pointplot(x=x, y=y, data=data.loc[::period])
    for ind, label in enumerate(ax.get_xticklabels()):
        if ind % period == 0:  # every 10th label is kept
            label.set_visible(True)
        else:
            label.set_visible(False)
    plt.legend()
    _ = plt.xticks(rotation=90)
