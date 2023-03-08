import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

loss_data = pd.read_csv("./train_val_loss.csv")
loss_data.columns = ["epoch", "loss", "type"]
print(loss_data)
sns.set(style="whitegrid")
sns.lineplot(x = "epoch", y = "loss", style = "type", hue = "type", data = loss_data)
plt.show() 