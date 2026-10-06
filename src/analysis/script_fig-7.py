import numpy as np

import matplotlib.pyplot as plt



# Five-seed matched 12-class results

models = [

    "RBF-SVM",

    "Linear SVM",

    "Random Forest",

    "XGBoost",

    "Fixed QK + SVC",

]



means = np.array([

    0.2680,

    0.2589,

    0.2392,

    0.2289,

    0.1498,

])



stds = np.array([

    0.0205,

    0.0147,

    0.0068,

    0.0131,

    0.0050,

])



# Same muted colors as the original Fig. 7

colors = [

    "#3E6685",

    "#78A3A5",

    "#A8C5BC",

    "#DAB66B",

    "#96786C",

]



# Typography

plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 11,

    "axes.labelsize": 12,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

})



fig, ax = plt.subplots(figsize=(7.2, 4.2))



x = np.arange(len(models))



bars = ax.bar(

    x,

    means,

    width=0.80,

    color=colors,

    edgecolor="black",

    linewidth=0.7,

    yerr=stds,

    capsize=4,

    error_kw={

        "ecolor": "black",

        "elinewidth": 1.0,

        "capthick": 1.0,

    },

)



# Axes

ax.set_ylabel("Test Macro-F1")

ax.set_xlabel("Model")



ax.set_xticks(x)

ax.set_xticklabels(

    models,

    rotation=25,

    ha="right",

)



ax.set_ylim(0, 0.35)

ax.set_yticks(np.arange(0, 0.351, 0.05))



# Match original light dotted horizontal grid

ax.grid(

    axis="y",

    linestyle=":",

    linewidth=0.7,

    alpha=0.45,

)



ax.set_axisbelow(True)



# Value labels

for bar, mean, sd in zip(bars, means, stds):

    ax.text(

        bar.get_x() + bar.get_width() / 2,

        mean + sd + 0.007,

        f"{mean:.3f}",

        ha="center",

        va="bottom",

        fontsize=10,

    )



# Same clean appearance

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)



fig.tight_layout()



# Publication versions

plt.savefig(

    "fig7_matched_12class_5seed.pdf",

    bbox_inches="tight",

)



plt.savefig(

    "fig7_matched_12class_5seed.png",

    dpi=600,

    bbox_inches="tight",

)



plt.show()
