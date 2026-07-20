from joblib import load

with open('vectorizer.joblib', 'rb') as f:
    vectorizer = load(f)
with open('clf.joblib', 'rb') as f:
    clf = load(f)

print("検証したい業界の概要文を入力してください。")

ken = str(input())
ken_list = [ken]

X_2 = vectorizer.transform(ken_list)
seikai = clf.predict(X_2)

print("予測結果:", seikai)
