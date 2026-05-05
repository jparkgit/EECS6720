import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Confusion matrix from your data
cm = np.array([
    [1420, 789],   # True -1
    [521,  2091]   # True +1
])

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[-1, 1])
disp.plot(cmap="Greys", colorbar=False)

plt.title("Confusion Matrix")
plt.show()