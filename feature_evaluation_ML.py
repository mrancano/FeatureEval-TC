#%%
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.svm import SVC
from scipy.stats import norm
import matplotlib.pyplot as plt

whichclass = 1

twenty_train = fetch_20newsgroups(subset='train')


categories = twenty_train.target_names


print(twenty_train.target)

print(twenty_train.target_names[whichclass])


count_vect = CountVectorizer(min_df=3)

X_train_counts = count_vect.fit_transform(twenty_train.data)
X_train_counts[X_train_counts!=0] = 1
print(np.shape(X_train_counts))

target_vec = twenty_train.target
target_vec[target_vec!=whichclass]=30
target_vec[target_vec==whichclass]=1
target_vec[target_vec==30] = 0

pos = np.sum(target_vec)
neg = len(target_vec)-pos


tp = np.array(np.sum(X_train_counts[target_vec==1,:],0))
fp = np.array(np.sum(X_train_counts[target_vec==0,:],0))
fn = pos-tp
tn = neg-fp

acc = tp-fp

tpr = tp/pos
fpr = fp/neg

tpr[tpr==0] = 0.0005
fpr[fpr==0] = 0.0005
tpr[tpr==1] = 1-0.0005
fpr[fpr==1] = 1-0.0005


acc2 = np.abs(tpr-fpr)

PR = tpr/fpr

BNS = np.abs(norm.ppf(tpr)-norm.ppf(fpr))

P_word = (tp+fp)/(pos+neg)
P_notword = 1-P_word

def e1(x,y):
    return -x/(x+y)*np.log2(x/(x+y))-y/(x+y)*np.log2(y/(x+y))

def e(x,y):
    # Avoid division by zero or log(0)
    mask = (x + y) != 0

    # Calculate entropy element-wise
    result = np.zeros_like(x, dtype=float)
    result[mask] = -x[mask] / (x[mask] + y[mask]) * np.log2(np.maximum(x[mask] / (x[mask] + y[mask]), 1e-10)) - y[mask] / (x[mask] + y[mask]) * np.log2(np.maximum(y[mask] / (x[mask] + y[mask]), 1e-10))

    return result


IG = e(pos,neg)-(P_word*e(tp,fp)+P_notword*e(fn,tn))

def gettopNfeatures(N,orderedfeatureMetric,sample):
    sorted_indices = sorted(range(len(orderedfeatureMetric[0])), key=lambda k: abs(orderedfeatureMetric[0][k]), reverse=True)

    top_N_indices = sorted_indices[:N]

    top_N_elements = [sample[0][0,i] for i in top_N_indices]
    
    return top_N_elements

def get_top_N_features_for_samples(N, ordered_feature_metric, samples):
    # Get the indices sorted by the absolute values of the feature metric array
    sorted_indices = np.argsort(ordered_feature_metric[0])[::-1]
    
    # Take the first N indices
    top_N_indices = sorted_indices[:N]
    
    # Use NumPy array indexing to get the elements at the top N indices for all samples
    results = samples[:, top_N_indices]

    return results

N = 3
feature_metric = [[-3, 7, -1, 5, -2, 4]]  # Replace this with your feature metric array
samples = np.array([
    [1, 2, 3, 4, 5, 6],
    [0, -1, 2, -3, 4, -5],
    [5, 4, 3, 2, 1, 0]
])  # Replace this with your array of samples

results = get_top_N_features_for_samples(N, feature_metric, samples)

print(f"Results for top {N} features for each sample:")
print(results)





#Train SVM on top N features as per a given Metric
metric = IG
def trainclassifier(N,metric = metric,raw_train=X_train_counts,target = target_vec,top = gettopNfeatures):
    #train_samples = np.zeros((np.shape(raw_train)[0],N))
    #for i,el in enumerate(raw_train):
       # print(i)
        #train_samples[i] = gettopNfeatures(N,metric,el)

    train_samples = get_top_N_features_for_samples(N,metric,raw_train)
    
    

    

    model = SVC()

    model.fit(train_samples,target)

    return model




twenty_test = fetch_20newsgroups(subset='test')
X_test_counts = count_vect.transform(twenty_test.data)
X_test_counts[X_test_counts!=0] = 1


test_vec = twenty_test.target
test_vec[test_vec!=whichclass]=0
test_vec[test_vec==whichclass]=1


def testclassifier(model,N,metric = metric,raw_test=X_test_counts,target=test_vec):
    test_samples = get_top_N_features_for_samples(N,metric,raw_test)
    print(np.shape(test_samples))
    pred_vec = []
    for text in test_samples:
        pred_vec.append(model.predict(text))
    pred_vec = np.array(pred_vec)
    error = pred_vec[:,0] -test_vec

    tp = sum((p == 1 and t == 1) for p, t in zip(pred_vec, test_vec))
    fp = sum((p == 1 and t == 0) for p, t in zip(pred_vec, test_vec))
    fn = sum((p == 0 and t == 1) for p, t in zip(pred_vec, test_vec))

    # Calculate precision, recall, and F1 score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f_measure = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f_measure

model = trainclassifier(1000,metric=acc)

print(testclassifier(model,1000,metric=acc))


N_list = [10,20,50,100,200,500,1000,2000]

F_list1 = []
F_list2 = []
F_list3 = []
F_list4 = []
for N in N_list:
    print(N)
    model1 = trainclassifier(N,metric=BNS)
    model2 = trainclassifier(N,metric=acc2)
    model3 = trainclassifier(N,metric=PR)
    model4 = trainclassifier(N,metric=IG)
    F_list1.append(testclassifier(model1,N,metric=BNS))
    F_list2.append(testclassifier(model2,N,metric=acc2))
    F_list3.append(testclassifier(model3,N,metric=PR))
    F_list4.append(testclassifier(model4,N,metric=IG))

plt.plot(N_list,F_list1,label="BNS")
plt.plot(N_list,F_list2,label="|Accuracy|")
plt.plot(N_list,F_list3,label="PR")
plt.plot(N_list,F_list4,label="IG")
plt.xlabel("number of features selected")
plt.ylabel("F-measure")
plt.legend()
plt.title("Skew 1:19 Model Performance for Various Feature Metrics")

    


# %%
plt.plot(N_list,F_list1,label="BNS")
plt.plot(N_list,F_list2,label="|Accuracy|")
plt.plot(N_list,F_list3,label="PR")
plt.plot(N_list,F_list4,label="IG")
plt.xlabel("number of features selected")
plt.ylabel("F-measure")
plt.legend()
plt.title("Skew 1:19 Model Performance for Various Feature Metrics")
# %%
