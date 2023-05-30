# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import cv2
import dataframe_image as dfi
from tqdm import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

df = pd.read_csv('Cosine_similarity_v2.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weights Model - Find Matching Items

# COMMAND ----------

def total_score(data, weights):
    feature_cols = ['image','prod','brand','color','comps','long']
    weighted_scores = []
    for i in tqdm(range(len(data))):
        scores = [data.loc[i,col]for col in feature_cols]
        weighted_score = round(sum(np.multiply(weights, scores)),2)
        weighted_scores.append(weighted_score)
    data['weighted_score']=weighted_scores

# COMMAND ----------

weights1 = [0.85, 0.1, 0, 0, 0.025, 0.025]
weights2 = [0.8, 0.1, 0, 0, 0.05, 0.05]
weights3 = [0.75, 0.15, 0, 0, 0.05, 0.05]
weights4 = [0.8, 0.08, 0, 0, 0.07,0.05]
weights5 = [0.82, 0.1, 0, 0, 0.05,0.03]
weights6 = [0.8, 0.1, 0, 0, 0.1,0]
weights7 = [0.82, 0.14, 0, 0, 0.02,0.02]
weights8 = [0.7, 0.2, 0,0,0.1,0]
weights9 = [0.8, 0.1, 0.1,0,0,0]
weights10 = [0.7, 0.17, 0.1,0.001,0.009,0.02]
weights11 = [0.75, 0.12, 0.1,0.001,0.009,0.02]
weights12 = [0.73, 0.119, 0.15,0.0001,0.0001,0.0008]
weights13 = [0.73, 0.10, 0.169,0.0001,0.0001,0.0008]
weights14 = [0.7, 0.03, 0.269,0.0001,0.0001,0.0008]
weights15 = [0.68, 0.01, 0.308,0.0005,0.0005,0.001]

# COMMAND ----------

total_score(df, weights15)

# COMMAND ----------

df.sort_values(by="weighted_score", inplace=True, ascending=False)

# COMMAND ----------

df1 = df.copy()

# COMMAND ----------

df_top = df1.groupby('lc')['weighted_score'].nlargest(1) 

# COMMAND ----------

df_res = df1.loc[df1.index.isin([i[1] for i in df_top.index])]

# COMMAND ----------

df_final = df_res.copy()
df_final = df_final.reset_index(drop=True)

# COMMAND ----------

def visual_pairs(data):
    for i in range(len(data)):
        lc_id = data['lc'][i]
        ff_id = str(data['ff'][i])
        image_filename1 = './lanecrawford_img/' + lc_id + '.jpg'
        image_filename2 = './farfetch_img/' + ff_id + '.jpg'
        img1 = cv2.imread(image_filename1)[:,:,(2,1,0)] 
        img2 = cv2.imread(image_filename2)[:,:,(2,1,0)]

        plt.subplot(121)
        plt.imshow(img1)
        plt.title(lc_id, fontsize=10)
        plt.axis('off')

        plt.subplot(122)
        plt.imshow(img2)
        plt.title(ff_id, fontsize=10)
        plt.axis('off')

        plt.subplots_adjust(wspace=1.5)

        plt.show()

# COMMAND ----------

# visual_pairs(df_final)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Algorithm -Radom Weights Tuning

# COMMAND ----------

# MAGIC %md
# MAGIC ##### According to the imbalanced, non decision-boundary characteristics of this dataset，random weights tuning is a more feasible and efficient approach

# COMMAND ----------

df_label = pd.read_excel('Label_Check.xlsx') #labeled data

# COMMAND ----------

df_label = df_label.iloc[:,0:9]
df_label = df_label.reset_index(drop=True)

# COMMAND ----------

def total_score(df, weights):
    feature_cols = ['image','prod','brand','color','comps','long']
    weighted_scores = []
    for i in range(len(df)):
        scores = [df.loc[i,col]for col in feature_cols]
        weighted_score = sum(np.multiply(weights, scores))
        weighted_scores.append(weighted_score)
    df['weighted_score']=weighted_scores

# COMMAND ----------

def generate_weight_vector(df, largest_no, iter_n):
    accuracy_ratios = []
    best_weight = None
    best_accuracy_ratio = 0
    TypeIerror = 0
    TypeIIerror = 0
    
    for i in tqdm(range(iter_n)):
        weight = np.random.dirichlet(np.ones(6), size=1)
        weights = weight[0].tolist()
        if (max(weights)==weights[0]) & (min(weights)==weights[3]):
            total_score(df, weights)
            matched_item = df.groupby('lc')['weighted_score'].nlargest(largest_no)
            df_match = df.loc[df.index.isin([i[1] for i in matched_item.index])]
            df_match['result'] = ['Y' for i in range(len(df_match['label']))]
            accuracy_ratio = len(df_match[df_match['label']==1])/len(df[df['label']==1])
            TypeIerror_ = (len(df_match[(df_match['result']=='Y') & (df_match['label']==-1)]))/len(df['label']==1)
            TypeIIerror_ = (len(df[df['label']==1])-len(df_match[df_match['label']==1]))/len(df['label']==1)
            accuracy_ratios.append(accuracy_ratio)
            if max(accuracy_ratios) <= best_accuracy_ratio:
                continue
            else:
                best_accuracy_ratio = max(accuracy_ratios)
                best_weight = weights
                TypeIerror = TypeIerror_
                TypeIIerror = TypeIIerror_
        else:
            continue
            
    
    return best_weight,best_accuracy_ratio,TypeIerror,TypeIIerror

# COMMAND ----------

generate_weight_vector(df_label, 1, 10000)

# COMMAND ----------

generate_weight_vector(df_label, 2, 10000)

# COMMAND ----------

generate_weight_vector(df_label, 3, 10000)

# COMMAND ----------

generate_weight_vector(df_label, 1, 300000)

# COMMAND ----------

generate_weight_vector(df_label, 2, 300000)

# COMMAND ----------

generate_weight_vector(df_label, 3, 300000)

# COMMAND ----------

generate_weight_vector(df_label, 3, 300000)

# COMMAND ----------

generate_weight_vector(df_label, 3, 500000)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Appendix

# COMMAND ----------

# MAGIC %md
# MAGIC #### Optimization of extremely unbalanced datasets with KNN+SMOTE 

# COMMAND ----------

df = df_label.copy()
df = df.iloc[:,2:9]
similarity_score = df.iloc[:,2:8]
inputs = similarity_score.values
label = df.loc[:,'label']
label.replace({-1:0},inplace=True)
labels = label.values

# COMMAND ----------

from imblearn.over_sampling import SMOTE
from imblearn.datasets import make_imbalance
from imblearn.over_sampling import RandomOverSampler 
from imblearn.over_sampling import BorderlineSMOTE 
from imblearn.over_sampling import SVMSMOTE 
from imblearn.over_sampling import KMeansSMOTE
from sklearn.cluster import KMeans
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors# k-nearest neighbor algorithm

# COMMAND ----------

class Smote:
    def __init__(self,samples,N,k):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0

    def over_sampling(self):
        N=int(self.N)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)    # 1. for each minority class sample, find its k-nearest neighbor in all minority class samples
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            self._populate(N,i,nnarray)
        return self.synthetic
    
    # 2. select N of the k nearest neighbors for each minority class sample
    # 3. generate N synthetic samples
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1

# COMMAND ----------

# Using SMOTE randomly generate new samples 
posDf = df[df['label'] == 1].drop(['label'], axis=1)    
posArray = posDf.values    
newPosArray = Smote(posArray, 10, 10).over_sampling() #Take 10 new sets of data from 10 neighboring clusters
newPosDf = pd.DataFrame(newPosArray)  

# COMMAND ----------

newPosDf.columns = posDf.columns    
newPosDf['label'] = 1   

# COMMAND ----------

newPosDf = newPosDf[:10000] 
data = pd.concat([df, newPosDf])

# COMMAND ----------

inputs = data.iloc[:,0:6]
inputs = inputs.values
label = data.loc[:,'label']
labels = label.values

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perceptron Model indicates linear inseparability of dataset

# COMMAND ----------

class Perceptron:
    def __init__(self, eta=1):
        self.eta = eta  
        self.w = None  
        self.b = None  

    def fit(self, X_data, y_data):
        self.w = np.zeros(X_data.shape[1])  
        self.b = 0
        change = True
        while change:  
            for X, y in zip(X_data, y_data):  
                change = False
                while y * (self.w @ X + self.b) <= 0:
                    self.w += self.eta * X * y
                    self.b += self.eta * y
                    change = True
        return

    def predict(self, X):
        return np.sign(self.w @ X + self.b)


class Perceptron_dual:
    # Perception in dyadic form
    def __init__(self, eta=1):
        self.eta = eta # when eta is 1 it is the number of times each sample is involved in the training
        self.b = None
        self.alpha = None  # alpha corresponds to the weight of the sample

        self.N = None
        self.gram = None

    def init_param(self, X_data):
        self.N = X_data.shape[0]
        self.alpha = np.zeros(self.N)
        self.b = 0
        self.gram = self.getGram(X_data)

    def getGram(self, X_data):
        # Calculate the Gram matrix
        gram = np.diag(np.linalg.norm(X_data, axis=1) ** 2)

        for i in range(self.N):
            for j in range(i + 1, self.N):
                gram[i, j] = X_data[i] @ X_data[j]
                gram[j, i] = gram[i, j]

        return gram

    def sum_dual(self, y_data, i):
        s = 0
        for j in range(self.N):
            s += self.alpha[j] * y_data[j] * self.gram[j][i]
        return y_data[i] * (s + self.b)

    def fit(self, X_data, y_data):
        self.init_param(X_data)
        changed = True
        while changed:
            changed = False
            for i in range(self.N):  
                while self.sum_dual(y_data, i) <= 0:
                    self.alpha[i] += self.eta
                    self.b += self.eta * y_data[i]
                    changed = True
        return


# COMMAND ----------

# the code is unstoppable
if __name__ == '__main__':
    X = inputs
    y = labels
    p = Perceptron()
    p.fit(X, y)     
    print(p.w, p.b)

# COMMAND ----------

# the code is unstoppable
if __name__ == '__main__':
    X = inputs
    y = labels
    p = Perceptron_dual()
    p.fit(X_data, y_data)
    print(p.w, p.b)

# COMMAND ----------

# MAGIC %md
# MAGIC ### LogisticClassifier indicates non linear decision-boundary

# COMMAND ----------

X = inputs
y = labels

# COMMAND ----------

class LogisticClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LogisticClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x_in, apply_softmax=False):
        a_1 = self.fc1(x_in)
        y_pred = self.fc2(a_1)
        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)
        return y_pred

# COMMAND ----------

model = LogisticClassifier(input_dim=args.dimensions, 
                           hidden_dim=args.num_hidden_units, 
                           output_dim=args.num_classes)
print (model.named_modules)

# COMMAND ----------

model = LogisticClassifier(input_dim=6, 
                           hidden_dim=10, 
                           output_dim=8)

# COMMAND ----------

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# COMMAND ----------

def get_accuracy(y_pred, y_target):
    n_correct = torch.eq(y_pred, y_target).sum().item()
    accuracy = n_correct / len(y_pred) * 100
    return accuracy

# COMMAND ----------

for t in range(args.num_epochs):
    # propagate forward
    y_pred = model(X_train)
    
    # accuracy
    _, predictions = y_pred.max(dim=1)
    accuracy = get_accuracy(y_pred=predictions.long(), y_target=y_train)

    # loss
    loss = loss_fn(y_pred, y_train)
    
    # verbose
    if t%20==0: 
        print ("epoch: {0:02d} | loss: {1:.4f} | acc: {2:.1f}%".format(
            t, loss, accuracy))

    # gradient to zero
    optimizer.zero_grad()

    # backpropagation
    loss.backward()

    # update weight
    optimizer.step()

# COMMAND ----------

# MAGIC %md
# MAGIC epoch: 00 | loss: 1.8384 | acc: 68.1%
# MAGIC
# MAGIC epoch: 20 | loss: 12.8809 | acc: 96.1%
# MAGIC
# MAGIC epoch: 40 | loss: 5.9578 | acc: 96.1%
# MAGIC
# MAGIC epoch: 60 | loss: 2.5559 | acc: 96.1%
# MAGIC
# MAGIC epoch: 80 | loss: 0.6116 | acc: 96.1%
# MAGIC
# MAGIC epoch: 100 | loss: 0.2654 | acc: 96.1%
# MAGIC
# MAGIC epoch: 120 | loss: 0.2271 | acc: 96.1%
# MAGIC
# MAGIC epoch: 140 | loss: 0.1470 | acc: 96.1%
# MAGIC
# MAGIC epoch: 160 | loss: 0.1149 | acc: 96.1%
# MAGIC
# MAGIC epoch: 180 | loss: 0.1114 | acc: 96.1%

# COMMAND ----------

_, pred_train = model(X_train, apply_softmax=True).max(dim=1)
_, pred_test = model(X_test, apply_softmax=True).max(dim=1)

# COMMAND ----------

train_acc = get_accuracy(y_pred=pred_train, y_target=y_train)
test_acc = get_accuracy(y_pred=pred_test, y_target=y_test)
print ("train acc: {0:.1f}%, test acc: {1:.1f}%".format(train_acc, test_acc))

# COMMAND ----------

# MAGIC %md
# MAGIC train acc: 96.1%, test acc: 96.3%

# COMMAND ----------

def plot_multiclass_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))
    cmap = plt.cm.Spectral
    
    X_test = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    y_pred = model(X_test, apply_softmax=True)
    _, y_pred = y_pred.max(dim=1)
    y_pred = y_pred.reshape(xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

# COMMAND ----------

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_multiclass_decision_boundary(model=LogisticClassifier(input_dim=2, 
                           hidden_dim=10, 
                           output_dim=8), X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_multiclass_decision_boundary(model=LogisticClassifier(input_dim=2, 
                           hidden_dim=10, 
                           output_dim=8), X=X_test, y=y_test)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### MLP Model indicates non decision-boundary

# COMMAND ----------

from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# COMMAND ----------

args = Namespace(
    seed=1000,
    num_samples_per_class=None,
    dimensions=2,
    num_classes=2,
    train_size=0.9,
    test_size=0.1,
    num_hidden_units=100,
    learning_rate=1e-0,
    regularization=1e-3,
    num_epochs=200,
)

np.random.seed(args.seed)

# COMMAND ----------

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).long()

# COMMAND ----------

shuffle_indicies = torch.LongTensor(random.sample(range(0, len(X)), len(X)))
X = X[shuffle_indicies]
y = y[shuffle_indicies]

test_start_idx = int(len(X) * args.train_size)
X_train = X[:test_start_idx] 
y_train = y[:test_start_idx] 
X_test = X[test_start_idx:] 
y_test = y[test_start_idx:]

# COMMAND ----------

# define the model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        a_1 = F.relu(self.fc1(x_in))
        y_pred = self.fc2(a_1)

        if apply_softmax:
            y_pred = F.softmax(y_pred, dim=1)

        return y_pred

model = MLP(input_dim=6, hidden_dim=10, output_dim=8)

# define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# train the model
for t in range(100):
    # forward pass
    y_pred = model(X_train)

    # compute loss and accuracy
    loss = loss_fn(y_pred, y_train)
    accuracy = (y_pred.argmax(dim=1) == y_train).float().mean()

    # backpropagation and optimizer steps
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print training progress
    if t % 10 == 0:
        print(f"Epoch {t}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")

# make predictions on the test set
with torch.no_grad():
    y_pred_test = model(X_test)
    test_accuracy = (y_pred_test.argmax(dim=1) == y_test).float().mean()

print(f"Test Accuracy: {test_accuracy:.4f}")


# COMMAND ----------

plt.figure(figsize=(12,5))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_multiclass_decision_boundary(model=MLP(input_dim=2, hidden_dim=10, output_dim=8), X=X_train, y=y_train)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_multiclass_decision_boundary(model=MLP(input_dim=2, hidden_dim=10, output_dim=8), X=X_test, y=y_test)
plt.show()