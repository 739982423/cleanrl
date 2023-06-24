import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

loss_data = pd.read_csv("./train_val_loss_new.csv",encoding='gb2312')
loss_data.columns = ["loss", "epoch", "type"]
print(loss_data)
sns.set(style="ticks")

mark = [19 * i + 8 for i in range(8)]
print(mark)
figure, ax = plt.subplots()
palette = sns.color_palette('colorblind')
sns.lineplot(x = "epoch", y = "loss", hue = "type", data = loss_data, dashes=False, linestyle="-", 
                markers='o', markersize=7, markeredgewidth=1, markevery=mark)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
font1 = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 14,
            }
font2 = {'family': 'STZhongsong',
            'weight': 'normal',
            'size': 14,
            }
plt.grid(linestyle='-.')
plt.tick_params(labelsize=13)
plt.grid(linestyle='-.')
plt.legend(prop=font2, loc=1, markerscale=1,)
plt.xlabel('训练次数', fontdict=font2)
plt.ylabel('均方误差（MSE）', fontdict=font2)
plt.tight_layout()
plt.show() 