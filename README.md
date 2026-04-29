# covid detection with CNNs

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org)

trained a convolutional neural network to spot COVID-19 in chest X-rays. three classes: covid positive, viral pneumonia, and normal lungs.

## what's in here

used the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle. built the model from scratch:

- a few convolutional layers with ReLU
- max pooling to shrink things down
- dropout so it doesn't memorize
- fully connected layers at the end
- SGD for optimization

the goal was to see how far you can get with a straightforward CNN on medical images. turns out, pretty far.

## run it

```bash
pip install tensorflow numpy pandas matplotlib opencv-python jupyter
jupyter notebook COVID.ipynb
```

## data

grab the dataset here: [Kaggle link](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database)

---

*built while learning how deep learning applies to healthcare — not a production model, but a solid learning project*
