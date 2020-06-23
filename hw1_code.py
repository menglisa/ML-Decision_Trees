#SHAGUN GUPTA, LISA MENG, REID PATTIS

import pandas as pd
import numpy as np
import math
from collections import defaultdict
import id3

################################################################################################
# PART 1: IMPLEMENTATION FROM SCRATCH

dt = pd.read_csv('./dt_data.txt', delimiter=',', )

dt.rename(columns={'(Occupied':'Occupied', ' Price':'Price', ' Music':'Music', ' Location':'Location', ' VIP':'VIP',' Favorite Beer': 'Favorite Beer', ' Enjoy)': 'Enjoy'}, inplace = True)

dt['Occupied'] = dt['Occupied'].str.extract(r'^[0-9][0-9]:\s(?P<Occupied>.*).*')

dt['Enjoy'] = dt['Enjoy'].apply(lambda x: x.rstrip(';'))

for i in dt.columns.to_list():
    dt[i] = dt[i].apply(lambda x: x.strip())

class node:
    
    att_name = None
    ig = None
    branch_val = None
    
    
    def __init__(self, att, branch_val, ig):
        
        self.branch_val = branch_val
        self.att_name = att
        self.ig = ig
        
    def is_leaf(self):
        
        return False
        
class leaf_node:
    
    label_val = None
    branch_val = None
    
    def __init__(self, label_val, branch_val):
        
        self.label_val, self.branch_val= label_val, branch_val


    def is_leaf(self):
        
        return True
    
class Tree:
    
    def __init__(self,dt):
        
        self.tree = defaultdict(list)
        self.initial_data = dt
        self.features = self.get_att_values(dt)
        
    def get_att_values(self, dt):
        
        d = defaultdict()
        for i in dt.drop('Enjoy',axis=1).columns.to_list():
            d[i]= list(dt[i].unique())
        
        return d
        
    def calculate_root(self, dt_sub, parent, branch_val):
        
        #additional branch
        if isinstance(dt_sub,str):
            
            leaf = leaf_node(dt_sub, branch_val)
            self.tree[parent].append(leaf)
            print('leaf node found', dt_sub, parent.att_name, branch_val)
            return
        
        #first check if no more attributes
        
        if len(dt_sub.columns.to_list()) == 1 and len(dt_sub['Enjoy'].unique()) != 1:
            
            d = dt_sub['Enjoy'].value_counts()
            d1 = []
            for i in d.keys():
                d1.append([i,d[i]])
            
            d1.sort(key= lambda x: x[1], reverse = True)
            
            leaf = leaf_node(d1[0][0], branch_val)
            self.tree[parent].append(leaf)
            print('leaf node found', dt_sub['Enjoy'].unique()[0], parent.att_name, branch_val)
            return 
        
        #first check if all label values are same
        if len(dt_sub['Enjoy'].unique()) == 1 or sum(dt.nunique())/len(dt.columns.to_list())==1:
            
            leaf = leaf_node(dt_sub['Enjoy'].unique()[0], branch_val)
            self.tree[parent].append(leaf)
            print('leaf node found', dt_sub['Enjoy'].unique()[0], parent.att_name, branch_val)
            return
    
            
            
        #If not decide root node
        entropy = 0
        vals = dict(dt_sub['Enjoy'].value_counts())
        for val in vals:
            p = vals[val]/dt_sub.shape[0]
            entropy += p * math.log2(p) * -1
            
        
        ig = []
        for i in dt_sub.columns.to_list(): #traverse through all columns
            if i != 'Enjoy':
                avg = 0
                for j in dt_sub[i].unique(): #traverse through all unique values of said column
                    temp = dt_sub[dt_sub[i] == j] #get dataset for a value
                    d = dict(temp['Enjoy'].value_counts()) #calculate label distribution
                    i_x = 0
                    for k in d:
                        p = d[k]/temp.shape[0]
                        i_x += p * math.log2(p) * -1
                    avg += i_x * temp.shape[0] / dt_sub.shape[0]

                ig.append([entropy - avg, i])
        
        ig.sort(key=lambda x: x[0], reverse = True)
        
            
        curr_root = node(ig[0][1], branch_val, ig[0][0])
        self.tree[parent].append(curr_root)

        for i in list(set(self.features[ig[0][1]]) - set(dt_sub[ig[0][1]].unique())):
            
            x = dt_sub['Enjoy'].value_counts().idxmax()
            self.calculate_root(x , curr_root, i)

        
        for i in dt_sub[ig[0][1]].unique():
            
            temp = dt_sub[dt_sub[ig[0][1]] == i]
            
            temp.drop(columns = [ig[0][1]], inplace = True)
            
            if parent != -1:
            
                print('att node found', curr_root.att_name, parent.att_name, branch_val)
                
            else:
                print('att node found', curr_root.att_name, branch_val)
                
                
            self.calculate_root(temp, curr_root, i)


    def test(self,d):
        
        root= self.tree[-1][0]
        while root.is_leaf() == False:
            for i in self.tree[root]:
                if i.branch_val == d[root.att_name]:
                    root = i
                    break
        return root.label_val
                
    def visualize(self, parent,indent,f):
        
            
        for i in self.tree[parent]:
            if i.is_leaf():
                f.write(indent)
                f.write('-')
                f.write(str(i.branch_val))
                f.write('->')
                f.write(str(i.label_val))
                f.write('\n')

            else:
                f.write(indent)
                f.write('-')
                f.write(str(i.branch_val))
                f.write('->')
                f.write(i.att_name)
                f.write('\n')
                self.visualize(i,indent + '\t',f)
                

home = Tree(dt)
print(home.features)
home.calculate_root(dt, -1, -1)
f = open('decision_trees.txt','w')
home.visualize(-1,'',f)
f.close()
d = {'Occupied' : 'Moderate', 'Price' : 'Cheap', 'Music' : 'Loud', 'Location' : 'City-Center', 'VIP' : 'No', 'Favorite Beer' : 'No'}
print(home.test(d))

################################################################################################   
#PART 2: IMPLEMENTATION THROUGH LIBRARY

df = pd.read_csv("dt_data.txt")
header_replacement = dict()
for header in df.columns.values:
    original = header
    header = header.replace("(","")
    header = header.replace(")","")
    header_replacement[original] = header.strip()
print(header_replacement)
df.rename(columns=header_replacement, inplace=True)
df["Occupied"] = df["Occupied"].apply(lambda s: s.split(":")[1])
df["Enjoy"] = df["Enjoy"].apply(lambda s: s[:-1])
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x) # removes whitespace
feature_names = list(df.columns.values)[:-1]
l = list(df.columns.values)[-1] # l = label
X = df.drop(l, axis=1).to_numpy()
exec("X=np." + repr(X)[:repr(X).rfind(",")] + ")")
y = df[l].to_numpy()
exec("y=np." + repr(y)[:repr(y).rfind(",")] + ")")
clf = id3.Id3Estimator()
clf.fit(X, y, check_input=True)
print(id3.export_text(clf.tree_, feature_names))
test_data = np.array( [ ["Moderate","Cheap","Loud","City-Center","No","No"] ] )
result = clf.predict(test_data)
print(result)   