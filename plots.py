from main import main
import numpy as np
import matplotlib.pyplot as plt

class Object:
  pass

def testKNNClassification():
  accuracies = [-1]
  macrof1s = [-1]
  for i in range(1, 30):
    np.random.seed(100)

    args = Object()
    args.task = 'breed_identifying'
    args.method = 'knn'
    args.K = i
    args.data_type = 'features'
    args.test = False

    print(f"\n### KNN with K = {i} ###")
    
    acc, macrof1 = main(args)
    accuracies.append(acc)
    macrof1s.append(macrof1)

  print(accuracies)
  print(macrof1s)

  print(f"\n\nBest K: {np.argmax(accuracies)} with accuracy: {accuracies[np.argmax(accuracies)]}")
  print(f"Best K: {np.argmax(macrof1s)} with F1 Score: {macrof1s[np.argmax(macrof1s)]}")


  plt.figure(1)
  plt.plot(range(1, 30), accuracies[1:], label='Accuracy', marker='o')
  plt.xlabel('K')
  plt.ylabel('Accuracy')
  plt.legend()

  plt.figure(2)
  plt.plot(range(1, 30), macrof1s[1:], label='F1 Score', marker='o')
  plt.xlabel('K')
  plt.ylabel('F1 Score')
  plt.legend()
  plt.show()

def testKNNRegression():
  losses = []
  for i in range(1, 50):
    np.random.seed(100)

    args = Object()
    args.task = 'center_locating'
    args.method = 'knn'
    args.K = i
    args.data_type = 'features'
    args.test = False

    print(f"\n### KNN with K = {i} ###")
    
    train_loss, loss = main(args)
    losses.append(loss)
  
  best = min(losses)
  print(losses)
  print(f"\n\nBest K: {np.argmin(losses)} with loss: {best}")

  plt.plot(range(1, 50), losses, label='Loss', marker='o')
  plt.xlabel('K')
  plt.ylabel('Loss')
  plt.legend()
  plt.show()


def testLogisticRegression():
  macrof1s = []
  for lr in range(0, 200, 3):
    adjusted_lr = lr / 1000 + 0.001
    for max_iter in range(10, 200, 10):
      np.random.seed(100)

      args = Object()
      args.task = 'breed_identifying'
      args.method = 'logistic_regression'
      args.lr = adjusted_lr
      args.max_iters = max_iter
      args.data_type = 'features'
      args.test = False

      print(f"\n### LOGISTIC REGRESSION with lr = {adjusted_lr}, max_iter = {max_iter} ###")
      
      acc, macrof1 = main(args)
      result = Object()
      result.lr = adjusted_lr
      result.max_iter = max_iter
      result.macrof1 = macrof1
      macrof1s.append(result)

  best = max(macrof1s, key=lambda e: e.macrof1)
  print(macrof1s)
  print(f"\n\nBest lr: {best.lr}, Best max_iter: {best.max_iter} with F1 Score: {best.macrof1}")

  x = [e.lr for e in macrof1s]
  y = [e.max_iter for e in macrof1s]
  z = [e.macrof1 for e in macrof1s]
  colors = [(e.macrof1 / best.macrof1, 0, 0) for e in macrof1s]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, c=colors, marker='o')

  ax.set_xlabel('Learning Rate')
  ax.set_ylabel('Max Iterations')
  ax.set_zlabel('F1 Score')

  plt.show()

def testLinearRegression():
  losses = []
  for lmda in range(0, 50, 1):
    lmda_adjusted = lmda / 10
    np.random.seed(100)

    args = Object()
    args.task = 'center_locating'
    args.method = 'linear_regression'
    args.lmda = lmda_adjusted
    args.data_type = 'features'
    args.test = False

    print(f"\n### LINEAR REGRESSION with lambda = {lmda_adjusted} ###")
    
    train_loss, loss = main(args)
    res = Object()
    res.lmda = lmda_adjusted
    res.loss = loss
    res.train_loss = train_loss
    losses.append(res)
  
  best = min(losses, key=lambda e: e.loss)
  print(losses)
  print(f"\n\nBest lambda: {best.lmda} with loss: {best.loss}")

  x = [e.train_loss for e in losses]
  y = [e.loss for e in losses]
  z = [e.lmda for e in losses]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, c="r", marker='o')

  ax.set_xlabel('Train Loss')
  ax.set_ylabel('Loss')
  ax.set_zlabel('Lambda')

  plt.show()


if __name__ == '__main__':
  # testKNNClassification()
  # testLogisticRegression()
  # testKNNRegression()
  testLinearRegression()