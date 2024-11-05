import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from config import DATASETS
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
tf.keras.utils.set_random_seed(42)
import time


d = DATASETS["NetworkSlicing5G"]()

X_train, y_train = d.load_training_data()
x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, random_state=42, test_size=0.2)

X_test, y_test = d.load_test_data()

callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=10)

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath="temp",
    monitor='accuracy',
    mode='max',
    save_best_only=True)

cls = d.create_model()
start = time.time()
history = cls.fit(x_train, y_train, batch_size=8192, epochs=200, verbose=0, callbacks=[callback, model_checkpoint_callback])
end = time.time()-start
print(end)
cls = tf.keras.models.load_model("temp")
y_pred = cls.predict(X_test, verbose=0)

y_pred = [np.argmax(y) for y in y_pred]

acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
mcc = matthews_corrcoef(y_test, y_pred)

print(acc, f1, mcc)
plt.plot(history.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()