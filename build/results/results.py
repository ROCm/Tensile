import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

for i in [100, 1000, 10000, 100000, 1000000]:

  print("Gathering results for cache = %d" % i)

  f_name = "cacheSize_" + str(i) + ".txt"
  f = open(f_name, "r")
  lines = f.readlines()
  data = [int(x) for x in lines]
  data.sort()
  
  top_n = 99
  length = len(data)
  r_idx = int(top_n/100 * length)
  top_n_data = data[:r_idx]
  
  mu, std = norm.fit(data)
  mu_top, std_top = norm.fit(top_n_data)
  
  plt.subplot(1,2,1)
  plt.hist(data, bins=31, density=True, alpha=0.6, color="g")
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu, std)
  plt.plot(x, p, 'k', linewidth=2)
  plt.title("mu, std = %.2f, %.2f" % (mu, std))
  
  plt.subplot(1,2,2)
  plt.hist(top_n_data, bins=31, density=True, alpha=0.6, color="g")
  xmin, xmax = plt.xlim()
  x = np.linspace(xmin, xmax, 100)
  p = norm.pdf(x, mu_top, std_top)
  plt.plot(x, p, 'k', linewidth=2)
  plt.title("Top %d%%: mu, std = %.2f, %.2f" % (top_n, mu_top, std_top))

  plt.suptitle("Cache Size = %d" % i)
  
  #plt.show()
  plt.savefig(f_name.replace("txt", "jpg"))
  plt.clf()
