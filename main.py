import streamlit as st
import pickle
import pandas as pd
import numpy as np
from PIL import Image
import gzip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

st.header('HR Analytics: AI Evaluation Of a Possible Job Change :robot_face:')

st.markdown('![](https://github.com/SovBakoid/HR/raw/main/bender-futurama.gif)')

st.subheader('Random forest full model design :chart_with_upwards_trend:')


with Image.open("model_design.png") as im:
    st.image(im)

@st.cache
def get_data():
    return pd.read_csv('aug_train.csv')

data=get_data()

with gzip.open('hohohaha.pklz', 'rb') as ifp:
    model=pickle.load(ifp)

st.subheader('Please, fill a form about yourself :pencil:')

city_development_index=st.slider('Pick your city development index', 0, 100)

city_development_index=city_development_index/100

gender=st.radio('Pick your gender', data['gender'].dropna().unique())

relevent_experience=st.radio('How much experince do you have', data['relevent_experience'].dropna().unique())

enrolled_university=st.radio('Pick your type of University course enrolled if any', data['enrolled_university'].dropna().unique())

education_level=st.radio('Pick your education level', data['education_level'].dropna().unique())

major_discipline=st.radio('Pick your education major discipline', data['major_discipline'].dropna().unique())

experience=st.radio('Pick your total experience in years', data['experience'].dropna().unique())

company_size=st.radio("Pick number of employees in current employer's company", data['company_size'].dropna().unique())

company_type=st.radio('Pick your current company type', data['company_type'].dropna().unique())

last_new_job=st.radio('Pick difference in years between previous job and current job', data['last_new_job'].dropna().unique())

training_hours=st.slider('Pick how many ours would you like to train', 0, 100)

submit=st.button('Submit')

class NeuralNet9(nn.Module):
    def __init__(self):
        super(NeuralNet9, self).__init__()
        self.lin1 = nn.Linear(in_features=26, out_features=100)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.3)
        self.lin2 = nn.Linear(in_features=100, out_features=100)
        self.relu2 = nn.ReLU()
        self.lin3 = nn.Linear(in_features=100, out_features=2)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.drop1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        return x

model9 = NeuralNet9()
model9.load_state_dict(torch.load('nn_model9'))


def feature_work(X, y):
    num = X.select_dtypes(exclude='object').columns

    cat = X.select_dtypes(include='object').columns

    impute = SimpleImputer(strategy='most_frequent')

    Xc = impute.fit_transform(X[cat])

    encoding = OneHotEncoder()

    Xc = encoding.fit_transform(Xc)

    Xc = Xc.toarray()

    transform = PowerTransformer()

    Xn = transform.fit_transform(X[num])

    fs = SelectKBest(score_func=chi2, k=24)

    fs.fit(Xc, y)

    Xfs = fs.transform(Xc)

    X = np.concatenate([Xn, Xfs], axis=1)

    return X, y

if submit:
    list_of_stuff=[city_development_index, gender, relevent_experience,
     enrolled_university, education_level, major_discipline,
     experience, company_size, company_type, last_new_job,
     training_hours]

    test_dictt={}

    for i in range(len(data.drop(columns=['enrollee_id', 'city', 'target']).columns)):
        test_dictt[data.drop(columns=['enrollee_id', 'city', 'target']).columns[i]]=[list_of_stuff[i]]

    test_x=pd.DataFrame(test_dictt)

    test_dictt['target']=[0]
    test_dictt['enrollee_id'] = [100000]
    test_dictt['city'] = ['city_67']

    test_XnY=pd.DataFrame(test_dictt)

    data1 = data.append(test_XnY, ignore_index=True)

    X, y = data1.drop(columns=['enrollee_id', 'city', 'target'], axis=1), data1['target']

    X_res, y_res = feature_work(X, y)

    chance_by_nn=F.softmax(model9(torch.Tensor(X_res[-1])), dim=1)[1]

    res=model.predict(test_x)

    if res:
        st.success(f'Congratulations. The random forest model does recommend you for enrollment and neural network gives you a {chance_by_nn} chance of staying in the company after finishing your training :heartbeat:')
    else:
        st.warning(f"Too bad. The random forest model doesn't recommend you for enrollment and neural network gives you a {chance_by_nn} chance of staying in the company after finishing your training :broken_heart:")