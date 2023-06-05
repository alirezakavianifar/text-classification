import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report


def calculate_accuracy():
    paths = r'D:\projects\text-classification\text-classification\repo\test_sabtenam4.xlsx'
    df = pd.read_excel(paths)
    count_plus = 0
    count_minus = 0

    for index, row in df.iterrows():
        if row['pred_labels'] == row['real_labels']:
            count_plus += 1
        else:
            count_minus += 1


def preprocess(text):
    return text[:5]

PATH_DIR = r'D:\projects\text-classification\text-classification\repo\sabtenam.xlsx'
dict_labels = {}

df_sabtenam = pd.read_excel(PATH_DIR, sheet_name='Sheet1')
df_sabtenam['gasht'] = df_sabtenam['کد پستی'].map(preprocess)
df_gasht = pd.read_excel(PATH_DIR, sheet_name='Sheet2')
df_gasht = df_gasht.astype('str')
df_sabtenam.dropna(inplace=True)
df_sabtenam = df_sabtenam.astype('str')
df_merged = df_sabtenam.merge(
    df_gasht, how='inner', left_on='gasht', right_on='گشت پستی')


labels = df_merged['کد اداره امور مالیاتی'].unique()


for index, item in enumerate(labels):
    dict_labels[item] = index

df_merged['edare_num'] = df_merged['کد اداره امور مالیاتی'].map(dict_labels)


X_train, X_test, y_train, y_test = train_test_split(df_merged['آدرس'],
                                                    df_merged['edare_num'],
                                                    test_size=0.2,
                                                    random_state=7)

clf = Pipeline([
    ('vectorizer_bow', CountVectorizer()),
    ('Multi NB', MultinomialNB())
])

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))

pred_labels = []
real_labels = []
for item in y_test:
    for key, value in dict_labels.items():
        if str(item) == str(value):
            pred_labels.append(key)

for item in y_pred:
    for key, value in dict_labels.items():
        if str(item) == str(value):
            real_labels.append(key)

X_test[:20].tolist()
y_test[:20].tolist()
y_pred[:20].tolist()

new_df = pd.DataFrame({'X_test': X_test, 'y_test': y_test, 'y_pred': y_pred,
                       'pred_labels': pred_labels, 'real_labels': real_labels})

new_df.to_excel(
    r'D:\projects\text-classification\text-classification\repo\test_sabtenam4.xlsx')

lst_labels = []
for key, value in dict_labels.items():
    lst_labels.append([key, value])

dict_labels_df = pd.DataFrame(lst_labels, columns=['key', 'value'])


y_test_ = y_test.merge()

df_merged = new_df.merge(dict_labels_df, how='left',
                         left_on='y_test', right_on='value')
