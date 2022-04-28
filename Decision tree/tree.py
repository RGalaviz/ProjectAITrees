
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from numpy.lib.shape_base import split
from sklearn.metrics import accuracy_score

class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        ''' constructor ''' 
        
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        
        # for leaf node
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''
        
        # initialize the root of the tree 
        self.root = None
        
        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        
    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree ''' 
        
        X, Y = dataset[:,:-1], dataset[:,-1]
        num_samples, num_features = np.shape(X)
        
        # split until stopping conditions are met
        if num_samples>=self.min_samples_split and curr_depth<=self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["info_gain"]>0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth+1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth+1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"], 
                            left_subtree, right_subtree, best_split["info_gain"])
        
        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)
    
    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''
        
        # dictionary to store the best split
        best_split = {}
        max_info_gain = -float("inf")
        
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left)>0 and len(dataset_right)>0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_info_gain = self.information_gain(y, left_y, right_y, "entropy")
                    # update the best split if needed
                    if curr_info_gain>max_info_gain:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["info_gain"] = curr_info_gain
                        max_info_gain = curr_info_gain
                        
        # return best split
        return best_split
    
    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''
        
        dataset_left = np.array([row for row in dataset if row[feature_index]<=threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index]>threshold])
        return dataset_left, dataset_right
    
    def information_gain(self, parent, l_child, r_child, mode="entropy"):
        ''' function to compute information gain '''
        
        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        gain = self.entropy(parent) - (weight_l*self.entropy(l_child) + weight_r*self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        ''' function to compute entropy '''
        
        class_labels = np.unique(y)
        entropy = 0
        for cls in class_labels:
            p_cls = len(y[y == cls]) / len(y)
            entropy += -p_cls * np.log2(p_cls)
        return entropy
    
        
    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''
        
        Y = list(Y)
        return max(Y, key=Y.count)
    

    
    def print_tree(self, tree=None, indent="-"):
        ''' function to print the tree '''
        
        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            columns = ["Index","Refractive Index","Sodium","Magnesium","Silicon","Potassium","Calcium","Barium","Iron"]
            print("X_"+columns[tree.feature_index], "<=", tree.threshold, "Information Gain", tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)
    
    def fit(self, X, Y):
        ''' function to train the tree '''
        
        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        ''' function to predict new dataset '''
        
        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions
    
    def make_prediction(self, x, tree):
        ''' function to predict a single data point '''
        
        if tree.value!=None: 
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val<=tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)




columns = ["Index","Refractive Index","Sodium","Magnesium","Aluminum","Silicon","Potassium","Calcium","Barium","Iron","Type of glass"]

data = pd.read_csv('./glass-1.csv',names=columns)

print(data.head())

#take only the not index columns

data = data[["Refractive Index","Sodium","Magnesium","Aluminum","Silicon","Potassium","Calcium","Barium","Iron","Type of glass"]]

print(data.head())

#shuffle our dataset in order to have a better way of splitting it in test set and training set

my_df = shuffle(data)

my_df.reset_index(inplace=True, drop=True)

print(my_df.head())

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.3, random_state=50)

#time to split it in test set and training set

# size =  int(my_df.shape[0])

# df_t = my_df.iloc[:int(size/2),:]

# print("--training---")

# print(df_t.head())

# df_t_x = df_t[["Refractive Index","Sodium","Magnesium","Silicon","Potassium","Calcium","Barium","Iron"]]
# df_t_y = df_t[["Type of glass"]]

# print("--testing---")

# df_tst = my_df.iloc[int(size/2):int(size/2 + size/4)]

# print(df_tst.head())

# print("--validation---")

# df_val = my_df.iloc[int(size/2 + size/4):,:]

# print(df_val.head())


classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=5)
classifier.fit(X_train,Y_train)
classifier.print_tree()

Prediction = classifier.predict(X_test)

print("Predictions for Test Set---\n", Prediction)


Y_pred = classifier.predict(X_test)

# [1.51711,14.23,0.00,2.08,73.36,0.00,8.62,1.67,0.00,]
print("Predict for: [1.51711,14.23,0.00,2.08,73.36,0.00,8.62,1.67,0.00,]")
print("Class predited:",classifier.predict([[1.51711,14.23,0.00,2.08,73.36,0.00,8.62,1.67,0.00]]))

print("Predict for: [1.52320,12.78,0.00,1.56,73.36,0.00,8.62,1.67,0.00]")
print("Class predited:", classifier.predict([[1.51711,12.78,0.00,1.56,73.36,0.00,10.06,1.67,0.00]]))
print("Accuracy:\n",accuracy_score(Y_test, Y_pred))