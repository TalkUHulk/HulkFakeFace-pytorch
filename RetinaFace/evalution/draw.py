import matplotlib.pyplot as plt
from sklearn import metrics

dataName = ["Resnet50", "Resnet50_Gender", "MobileNetV2"]
file = ["../outputs/Resnet50/DiscROC.txt",
        "../outputs/Resnet50_Gender/DiscROC.txt",
        "../outputs/MobileNetV2/DiscROC.txt"]
color = ["r", "g", "b"]
xdata = [[], [], []]
ydata = [[], [], []]
linestyle = ['-', '--', '-.']
for i in range(len(file)):
    with open(file[i], "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            xdata[i].append(float(line[1]))
            ydata[i].append(float(line[0]))


plt.figure(1)
plt.title('ROC Curve', fontsize=15)
plt.xlabel("False Positive Rate", fontsize=15)
plt.ylabel("True Positive Rate", fontsize=15)
plt.xlim(0, 3000)
plt.ylim(0, 1)

plt.grid()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
for i in range(len(dataName)):
    #auc = metrics.auc(xdata[i], ydata[i])
    plt.plot(xdata[i][:25000], ydata[i][:25000], color=color[i], linewidth=2.0,
             linestyle=linestyle[i],
             label=dataName[i])
plt.legend(loc='lower right', fontsize=10)  # 标签位置
# plt.show()
plt.savefig('ROC.png', dpi=800)
