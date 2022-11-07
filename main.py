import streamlit as st
import pickle
import pandas as pd
from PIL import Image
import gzip

st.header('HR Analytics: AI Evaluation Of a Possible Job Change :robot_face:')

st.markdown('![](https://github.com/SovBakoid/HR/raw/main/bender-futurama.gif)')

st.subheader('Model design :chart_with_upwards_trend:')


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

if submit:
    list_of_stuff=[city_development_index, gender, relevent_experience,
     enrolled_university, education_level, major_discipline,
     experience, company_size, company_type, last_new_job,
     training_hours]

    test_dictt={}

    for i in range(len(data.drop(columns=['enrollee_id', 'city', 'target']).columns)):
        test_dictt[data.drop(columns=['enrollee_id', 'city', 'target']).columns[i]]=[list_of_stuff[i]]

    test_x=pd.DataFrame(test_dictt)

    res=model.predict(test_x)

    if res:
        st.success('Congratulations. The model does recommend you for enrollment :heartbeat:')
    else:
        st.warning("Too bad. The model doesn't recommend you for enrollment :broken_heart:")