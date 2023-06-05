import pandas as pd
import re
from sklearn.model_selection import train_test_split
import fasttext

PATH_DIR = r'D:\projects\text-classification\text-classification\repo\گشت پستی استان.xlsx'


def preprocess(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip().lower()
    return text


def get_data():
    df = pd.read_excel(PATH_DIR)
    df = df.astype('str')
    df.dropna(inplace=True)
    # df[0].replace('Clothing & Accessories',
    #               'Clothing_Accessories', inplace=True)
    df['کد اداره امور مالیاتی'] = "__label__" + df['کد اداره امور مالیاتی']

    df['concat'] = df['کد اداره امور مالیاتی'] + \
        ' ' + df['نام مناطق و خیابانهای اصلی']
        
    df['concat'] = df['concat'].map(preprocess)
    df.reset_index(drop=True, inplace=True)
    return df


if __name__ == '__main__':

    df = get_data()

    to_csv = True
    # df = pd.DataFrame({'data':['__label__1604 دشت آزادگان','__label__1606 خرمشهر']})
    df.to_csv(r'D:\projects\text-classification\text-classification\repo\testdata.csv',
              index=False, header=False)
    train, test = train_test_split(df['concat'], test_size=0.2)

    if to_csv:
        train.iloc[-1::].to_csv(r'D:\projects\text-classification\text-classification\repo\train.train.csv', columns=df['concat'], index=False,
                     header=False)
        test.to_csv(r'D:\projects\text-classification\text-classification\repo\test.test.csv',
                    columns=df['concat'], index=False, header=False)

    model = fasttext.train_supervised(
        input=train)

    model.test(
        r"D:\projects\text-classification\text-classification\repo\ecommerce.test.csv")
