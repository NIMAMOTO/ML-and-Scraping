import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from joblib import dump

ken = ("検証用データ.csv")
stu =  ('学習用データ.csv')
df1 = pd.read_csv(stu)
df2 = pd.read_csv(ken)

X = df1["概要文"]
y = df1["業界"]
X_2 = df2["概要文"]
y_2 = df2["業界"]

#事前処理を行う
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
X = vectorizer.fit_transform(X)
X_2 = vectorizer.transform(X_2)

#学習(fit)を行う
clf = LogisticRegression().fit(X, y)

#予測(predict)を行う
seikai = clf.predict(X_2)
dict_ = {'予測': seikai, '正解': y_2, '概要文': df2["概要文"]}
df = pd.DataFrame(dict_)
df = df.assign(予測結果= df['予測']== df['正解'])
df = df[['予測結果', '予測', '正解', '概要文']]

df.to_csv("predictions.csv", index=True, encoding="utf-8-sig")

print(df['予測結果'].mean())


with open('vectorizer.joblib', 'wb') as f:
    dump(vectorizer, f)
with open('clf.joblib', 'wb') as f:
    dump(clf, f)
