import matplotlib.pyplot as plt
import pandas as pd

results_path = 'runs/detect/train/results.csv'

results = pd.read_csv(results_path)

plt.figure()
plt.plot(results['                  epoch'], results['         train/box_loss'], label='train loss')
plt.plot(results['                  epoch'], results['           val/box_loss'], label='val loss', c = 'red')
plt.grid()
plt.title('Loss vs epochs')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()

plt.figure()
plt.plot(results['                  epoch'], results['   metrics/precision(B)'] * 100)
plt.grid()
plt.title('Validation accuracy vs epochs')
plt.ylabel('accuracy (%)')
plt.xlabel('epochs')

plt.show()