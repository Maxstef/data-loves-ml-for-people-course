import matplotlib.pyplot as plt
import seaborn as sns
import warnings

def dist_box_v0(dataset, column):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      plt.figure(figsize=(16,6))

      plt.subplot(1,2,1)
      sns.distplot(dataset[column], color = 'purple')
      pltname = 'Графік розподілу для ' + column
      plt.ticklabel_format(style='plain', axis='x')
      plt.title(pltname)

      plt.subplot(1,2,2)
      red_diamond = dict(markerfacecolor='r', marker='D')
      sns.boxplot(y = column, data = dataset, flierprops = red_diamond)
      pltname = 'Боксплот для ' + column
      plt.title(pltname)

      plt.show()

def draw_dist_box(dataset, column, bins=30, bw_adjust=0.75):
    plt.figure(figsize=(16, 6))

    # Distribution plot (replacement for distplot)
    plt.subplot(1, 2, 1)
    sns.histplot(
        data=dataset,
        x=column,
        kde=True,
        bins=bins,
        kde_kws={
            'bw_adjust': bw_adjust # >1 smoother, <1 more detailed
        },
        stat="density",
        color='purple'
    )
    plt.ticklabel_format(style='plain', axis='x')
    plt.title('Distribution plot for ' + column)

    # Boxplot
    plt.subplot(1, 2, 2)
    red_diamond = dict(markerfacecolor='r', marker='D')
    sns.boxplot(
        y=column,
        data=dataset,
        flierprops=red_diamond
    )
    plt.title('Boxplot for ' + column)

    plt.tight_layout()
    plt.show()
