import numpy as np
import pandas as pd
from kneed import KneeLocator

def pick_features(X,std):
    keep_columns = std >= 0.01 # 刪除標準差小於0.01的特徵
    return X.loc[:, keep_columns]

def z_score(X): # 標準化
    avg = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    z_score = (X - avg) / std
    return z_score, avg, std

def z_scorefor_test(X, avg, std):
    return (X - avg) / std

def pick_initial_center_random(X, k, random_state=42):# 隨機選擇k個資料點作為初始中心
    np.random.seed(random_state)
    indices = np.random.choice(X.shape[0], k, replace=False)
    center = X[indices]
    return center

def pick_initial_center_plus(x,k,random_state=42): # k means++ 版本 確保隨機取點不會選到太近的
    np.random.seed(random_state)
    n_samples = x.shape[0]
    
    centers = []
    chosen = []
    first_index = np.random.choice(n_samples)
    centers.append(x[first_index])
    chosen.append(chosen)
    
    for _ in range(1,k):
        center_array = np.array(centers)
        
        dist = compute_dist(x,center_array)
        
        min_dist = np.min(dist,axis=1) ** 2 
        
        if np.sum(min_dist) == 0:
            index = np.random.choice(n_samples)
            while(index in chosen):
                index = np.random.choice(n_samples)
        else:
            probability = min_dist / np.sum(min_dist)
            index = np.random.choice(n_samples, p = probability) #根據距離調整備選到的機率
            while(index in chosen):
                index = np.random.choice(n_samples, p = probability)
                
        centers.append(x[index])
        chosen.append(index)
    
    return np.array(centers)
    

from scipy.spatial.distance import cdist

def compute_dist(X, center):
    return cdist(X, center)

def assign_cluster(dist):
    cluster = np.argmin(dist,axis=1)
    return cluster

def update_center(X, cluster, k):
    center = np.zeros((k, X.shape[1]))
    for cluster_id in range(k):
        cluster_points = X[cluster == cluster_id]
        if len(cluster_points) > 0:
            center[cluster_id] = cluster_points.mean(axis=0) # 更新中心為該群集的平均值
        else:
            random_index = np.random.choice(X.shape[0])
            center[cluster_id] = X[random_index]
    return center

def k_means(X, k, select, max_iters=100, random_state=42):
    if select == "k means":
        center = pick_initial_center_random(X, k ,random_state)
    elif select == "k means++":
        center = pick_initial_center_plus(X, k, random_state)
    
    for i in range(max_iters):
        old_center = center.copy()
        
        dist = compute_dist(X, center)
        cluster = assign_cluster(dist)
        center = update_center(X, cluster, k)
        
        shift = np.sqrt(np.sum((center - old_center)**2))
        if shift < 1e-4: # 如果中心的移動小於閾值，則停止迭代
            break
        
    intertia = compute_intertia(X,cluster,center)
    return cluster, center, intertia

def predict_cluster(X, center):# 預測資料點的群集
    dist = compute_dist(X, center)
    cluster = assign_cluster(dist)
    min_dist = dist.min(axis=1)
    return cluster , min_dist

def compute_intertia(x, cluster, center): #計算intertia
    intertia = 0
    for cluster_id in range(center.shape[0]):
        cluster_point = x[cluster == cluster_id]
        
        if(len(cluster_point)) > 0:
            intertia += np.sum((cluster_point - center[cluster_id])**2)
    return intertia

def find_best_k_by_elbow(x, min_k=2, max_k=7, random_state=42):
    K_values = np.arange(min_k, max_k+1)
    all_interia=[]
    for k in range(min_k,max_k+1):
        temp_cluster, temp_center, interia = k_means(x, k, "k means++")
        all_interia.append(interia)
        
    all_interia = np.array(all_interia)
    
    l1 = KneeLocator(K_values, all_interia, curve="convex", direction="decreasing")
    
    top_2_elbow = np.argsort(l1.y_difference)[-2:] #紀錄top2的elbow候選
    top_2_elbow = top_2_elbow[::-1]
    top_2_k=[]
    top_2_k.append(K_values[top_2_elbow[0]]) 
    top_2_k.append(K_values[top_2_elbow[1]])
    return top_2_k

'''
train_data = pd.read_csv('dry_bean_train.csv') # 讀訓練資料並做對應的處理
X_train = train_data.drop('Class', axis=1)
Y_train = train_data['Class']
X_train, train_avg, train_std = z_score(X_train)
X_train = pick_features(X_train, train_std)
k = 5
train_cluster, train_center = k_means(X_train.values, k) # 訓練KMeans模型

table = pd.crosstab(train_cluster, Y_train)
print(table)
cluster_to_class={}

for i in range(5):
    most_one = table.loc[i].idxmax()
    cluster_to_class[i] = most_one

train_pred, train_min_dist = predict_cluster(X_train.values, train_center)
thresold = np.percentile(train_min_dist, 95) # 設定閾值為95百分位數
    
    
test_data = pd.read_csv('dry_bean_test.csv')
X_test = test_data.drop('Class', axis=1)
Y_test = test_data['Class']
X_test = pick_features(z_scorefor_test(X_test, train_avg, train_std), train_std)

test_pred, test_min_dist = predict_cluster(X_test.values, train_center)
Is_unknown = test_min_dist > thresold


pred_labels=[]
for i in range(len(X_test)):
    if Is_unknown[i]:
        pred_labels.append('unknown')
    else:
        pred_labels.append(cluster_to_class[test_pred[i]])
        
test_data['Final_Predicted'] = pred_labels
        
unknown_data = X_test[Is_unknown]
Y_test_unknown = Y_test[Is_unknown]
unknown_labels={}

if(len(unknown_data) > 0):# 如果有未知類別的資料，則對這些資料進行 KMeans 分群
    unknown_cluster, unknown_center = k_means(unknown_data.values, 2)
    temp = pd.crosstab(unknown_cluster, Y_test_unknown)
    print("\nUnknown cluster table:")
    print(temp)
    for c in range(2):
        most_one = temp.loc[c].idxmax()
        unknown_labels[c] = most_one
        
    new_unknown_pred=[]
    for i in range(len(unknown_data)):
        index = unknown_cluster[i]
        new_unknown_pred.append(unknown_labels[index])
    test_data.loc[Is_unknown, "Final_Predicted"] = new_unknown_pred
    
result = test_data["Final_Predicted"]
count = 0

for i in range(len(result)):
    if result[i] == Y_test[i]:
        count += 1

accuracy = count / (len(result))

print("\nKnown/Unknown Accuracy:", accuracy)
'''