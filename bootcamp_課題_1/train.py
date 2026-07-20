import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import Memory
from joblib import dump

memory = Memory("./bootcamp_kadai_1", )

ken = ('/Users/jianghaoyuan/Library/CloudStorage/OneDrive-関西学院/01_Projects/ML/bootcamp課題３＿各種データ - 検証用データ.csv')
stu =  ('/Users/jianghaoyuan/Library/CloudStorage/OneDrive-関西学院/01_Projects/ML/bootcamp課題３＿各種データ - 学習用データ.csv')
df1 = pd.read_csv(stu)
df2 = pd.read_csv(ken)
#print(df1.info())
X = df1["概要文"]
y = df1["業界"]
X_2 = df2["概要文"]
y_2 = df2["業界"]

#事前処理を行う
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(X)
X_2 = vectorizer.transform(X_2)
#feature_names = vectorizer.get_feature_names_out()
#print(feature_names[-700:])

#学習(fit)を行う
clf = LogisticRegression().fit(X, y)

#clf.predict(X[:2, :])

#予測(predict)を行う
#print(clf.score(X_2, y_2))
#print('X_2 =', X_2.shape, 'X =', X.shape)

seikai = clf.predict(X_2)
dict = {'予測': seikai, '正解': y_2, '概要文': df2["概要文"]}
df = pd.DataFrame(dict)
df = df.assign(予測結果= df['予測']== df['正解'])
df = df[['予測結果', '予測', '正解', '概要文']]

df.to_csv("predictions.csv", index=True, encoding="utf-8-sig")

print(df['予測結果'].mean())


with open('vectorizer.joblib', 'wb') as f:
    dump(vectorizer, f)
with open('clf.joblib', 'wb') as f:
    dump(clf, f)
