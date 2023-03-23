import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="whitegrid")
sns.set_palette('husl')
# Load an example dataset with long-form data
fmri = sns.load_dataset("fmri")

# flatui = ["#414B87", "#9CB3D4", "#ECEDFF", "#AF5A76"]
# sns.set_palette(flatui)
# sns.palplot(sns.color_palette())

sns.palplot(sns.color_palette("RdYlBu", 2))


# current_palette = sns.color_palette()
# sns.palplot(current_palette)

# Plot the responses for different events and regions
sns.lineplot(x="timepoint", y="signal",
             hue="region", style="event",
             data=fmri, palette=sns.color_palette("RdYlBu", 2))

plt.show()
