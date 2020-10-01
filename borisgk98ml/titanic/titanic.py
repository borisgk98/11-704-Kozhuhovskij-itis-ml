import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

surv = pd.read_csv("gender_submission.csv").Survived
ages = pd.read_csv("test.csv").Age

sns.kdeplot(ages[surv == 0], color="salmon", shade=True, alpha=0.15)
sns.kdeplot(ages[surv == 1], color="aqua", shade=True, alpha=0.15)
plt.legend(['Survived', 'Died'])
plt.show()

plt.hist(ages, density=True, facecolor='g', alpha=0.75)
plt.xlabel('Age')
plt.ylabel('Probability')
plt.grid(True)
plt.show()
