import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor as XGBR

from sklearn import svm
import time

#ngram参数
n = 3

def get_data(is_train_data):
    train_data = pd.read_csv('./data/data.csv',names=["url","status"])
    test_data = pd.read_csv('./data/data2.csv',names=["url","status"])
    train_X = train_data["url"]
    train_Y = train_data["status"]
    test_X = test_data["url"]
    test_Y = test_data["status"]
    train_y = []
    test_y = []
    for status in train_data["status"]:
        if "good" in status:
            train_y.append(1)
        else:
            train_y.append(0)
    for status in test_data["status"]:
        if "good" in status:
            test_y.append(1)
        else:
            test_y.append(0)
    if is_train_data == True:
        return train_X,train_y
    else:
        return test_X,test_y

#采用滑动窗口的方法分割数据
def get_ngrams(url):
    url_str = str(url)
    ngrams = []
    for i in range(0,len(url_str)-n):
        ngrams.append(url_str[i:i+n])
    return ngrams


class BaseModel(object):
    def __init__(self):
        pass
    
    def train(self):
        train_X,train_y = get_data(is_train_data = True)
        #数据向量化处理
        #tf-idf模型
        train_x = self.tf_idf_url(train_X)
        print("打印前5条数据: ")
        print(train_x[0:5])
        print(train_y[0:5])
        #分割训练集，训练模型
        #这里采用90%的数据作为训练集，10%的数据作为验证集
        x_train,x_test,y_train,y_test = train_test_split(train_x,train_y,test_size=0.1,random_state=0)
        print("开始训练...")
        start_t = time.time()
        self.classifier.fit(x_train,y_train)
        end_t = time.time()
        print("模型训练分数: {}".format(self.classifier.score(x_test,y_test)))
        print("训练时间: {}s".format(end_t - start_t))

    def predict(self):
        test_X,test_y = get_data(is_train_data = False)
        test_x = self.vectorlizer.transform(test_X)
        print("向量化后的维度是:"+str(test_x.shape))
        print("打印前5条数据: ")
        print(test_x[0:5])
        print(test_y[0:5])
        start_t = time.time()
        res = self.classifier.predict(test_x)
        score = 0
        for i in range(len(res)):
            if test_y[i] == res[i]:
                score=score+1
        end_t = time.time()
        print("测试集得分score：{}".format(score/len(res)))
        print("预测时间: {}s".format(end_t - start_t))

    def tf_idf_url(self,urls):
        train_x = self.vectorlizer.fit_transform(urls)
        print("向量化后的维度是:"+str(train_x.shape))
        return train_x


class LG(BaseModel):
    def __init__(self):
        self.vectorlizer = TfidfVectorizer(tokenizer = get_ngrams)
        self.classifier = LogisticRegression(max_iter = 10000)

class SVM(BaseModel):
    def __init__(self):
        self.vectorlizer = TfidfVectorizer(tokenizer = get_ngrams)
        self.classifier = svm.SVC(max_iter = 10000)

class RandomForest(BaseModel):
    def __init__(self):
        self.vectorlizer = TfidfVectorizer(tokenizer = get_ngrams)
        self.classifier = RandomForestClassifier()


model = LG()
# model = SVM()
# model = RandomForest()
model.train()
model.predict()

