
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, StackingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error 

#-----------------------#
# Page Layout

st.set_page_config(layout="wide")

#Title
st.title('Cost To Hospital Prediction')
image = Image.open('Hack.jpg')

st.image(image, width =600)



st.markdown ("""
            This app predicts Hospital approximate cost details based on patient admission time details
        
             """)

#About

expander_bar = st.beta_expander("INPUT FIELDS Details")
expander_bar.markdown("""
* **AGE**	Age of the patient
* **GENDER** 	Gender code for patient 
* **MARITAL STATUS** 	Marital Status of the patient:
* **KEY COMPLAINTS CODE**	Codes given to the key complaints faced by the patient:
* **BODY WEIGHT**	Weight of the patient
* **BODY HEIGHT**	Height of the patient
* **HR PULSE	**  Pulse of patient at the time of admission
* **BP-HIGH**	High BP of patient (Systolic)
* **BP-LOW** 	Low BP of patient (Diastolic)
* **RR**	Respiratory rate of patient
* **PAST MEDICAL HISTORY CODE**	Code given to the past medical history of the patient:
* **HB**	Hemoglobin count of patient
* **UREA**	Urea levels of patient
* **CREATININE** 	Creatinine levels of patient
* **MODE OF ARRIVAL**	Way in which the patient arrived the hospital:
* **STATE AT THE TIME OF ARRIVAL** 	State in which the patient arrived:
* **TYPE OF ADMISSION**	Type of admission for the patient:
* **TOTAL LENGTH OF STAY	** Number of days patient stayed in the hospital
* **LENGTH OF STAY-ICU** 	Number of days patient stayed in the ICU
* **LENGTH OF STAY-WARD**	Number of days patient stayed in the ward
* **IMPLANT USED (Y/N)** 	Any implant done on the patient
* **COST OF IMPLANT**	Total cost of all the implants done on the patient if any
                      """)
#---------------------------------------------------#

#Page Layout

col1 = st.sidebar
col2, col3 = st.beta_columns((1,1))


#---------------------------------------------#

col1.header('Input Options')

## Selectbox

sel_age = col1.slider('Age of the Patient', 0, 120, 30)
sel_gender = col1.selectbox(' Select Gender of Patient', ('Male', 'Female'))
sel_Married = col1.selectbox(' Select Marital Status', ('Married', 'Unmarried'))
sel_key_comp = col1.selectbox(' Select Key Compliance Code', ('ACHD', 'CAD-DVD', 'CAD-SVD', 'CAD-TVD', 'CAD-VSD', 'OS-ASD', 'other-heart', 'other-respiratory', 'other-general', 'other-nervous', 'other-tertalogy', 'PM-VSD', 'RHD'))
sel_weight = col1.slider('Patient Weight in Kg', 0.0, 150.0, 50.0, 0.5)
sel_height = col1.slider('Patient Height in cm', 0, 250, 145)
sel_pulse = col1.slider('Patient Pulse at time of Admission', 0, 200, 90)
sel_bphigh = col1.slider('Patient High BP Value', 50, 250, 110)
sel_bplow = col1.slider('Patient Low BP Value', 20, 150, 80)
sel_rr = col1.slider('Respiratory rate of patient', 10, 60, 25)
sel_past_hist = col1.selectbox('Select Past History code', ('Diabetes1', 'Diabetes2', 'Hypertension1', 'Hypertension2', 'Hypertension3', 'other'))
sel_arrival = col1.selectbox('Select Mode of Arrival', ('Ambulance', 'Walked In', 'Transferred'))
sel_admission = col1.selectbox('Type of Admission',('Elective','Emergency'))
sel_state = col1.selectbox('Select State in which patient Arrived', ('Alert', 'Confused'))
sel_HB = col1.slider('Hemoglobin count of Patient', 0, 30, 13)
sel_urea = col1.slider('Urea Levels of the patient', 1, 150, 22)
sel_cre = col1.slider('Creatinine levels of patient', 0.1, 6.0, 1.0, 0.1)
sel_stay = col1.slider('Number of days patient stayed in the hospital', 0, 50, 2)
sel_icu = col1.slider('Number of days patient stayed in the ICU', 0, 50, 2)
sel_ward = col1.slider('Number of days patient stayed in the ward', 0, 50, 2)
sel_imp = col1.slider('Total cost of all the implants done on the patient if any', 0, 100000, 1000, 100)
sel_alg = col1.radio( "Choose Your Algorithm for Prediction", 
                         ('LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'KNearest Neighbors', 'RandomForest', 'XGBoost', 'DecisionTree', 'Voting Regressor', 'Stacking Algorithm'))

# Formating Input Data For Prediction.

if sel_gender == 'Male':
    gender = 1
else:
    gender = 0

if sel_Married == 'Married':
    mar = 1
else:
    mar = 0
    
if sel_arrival == 'Ambulance':
    arr = 2
elif sel_arrival == 'Walked In':
    arr = 1
else:
    arr = 3

if sel_state == 'Alert':
    state = 1
else:
    state = 0

if sel_admission == 'Elective':
    adm = 0
else:
    adm = 1

if sel_past_hist == 'Diabetes1':
   dia1 = 1
else:
  dia1  = 0

if sel_past_hist == 'Diabetes2':
   dia2 = 1
else:
  dia2  = 0
  
if sel_past_hist == 'Hypertension1':
   hyp1 = 1
else:
  hyp1  = 0

if sel_past_hist == 'Hypertension2':
   hyp2 = 1
else:
  hyp2  = 0

if sel_past_hist == 'Hypertension3':
   hyp3 = 1
else:
  hyp3  = 0

if sel_past_hist == 'other':
   oth = 1
else:
  oth  = 0

if sel_key_comp  == 'ACHD':
   ach = 1
else:
  ach  = 0

if sel_key_comp  == 'CAD-DVD':
   cdvd = 1
else:
  cdvd = 0

if sel_key_comp  == 'CAD-SVD':
   csvd = 1
else:
  csvd = 0

if sel_key_comp  == 'CAD-TVD':
   ctvd = 1
else:
  ctvd = 0

if sel_key_comp  == 'CAD-VSD':
   cvsd = 1
else:
  cvsd = 0

if sel_key_comp  == 'OS-ASD':
   oasd = 1
else:
  oasd = 0

if sel_key_comp  == 'PM-VSD':
   pvsd = 1
else:
  pvsd = 0
  
if sel_key_comp  == 'RHD':
   rhd = 1
else:
  rhd = 0

if sel_key_comp  == 'other-heart':
   heart = 1
else:
  heart = 0

if sel_key_comp  == 'other-respiratory':
   resp = 1
else:
    resp = 0

if sel_key_comp  == 'other-general':
   gen = 1
else:
    gen = 0

if sel_key_comp  == 'other-nervous':
   ner = 1
else:
    ner = 0

if sel_key_comp  == 'other-tertalogy':
   ter = 1
else:
    ter = 0
    
      
data = {'AGE': sel_age,
            'GENDER': gender,
            'MARITAL_STATUS': mar,
            'BODY_WEIGHT': sel_weight,
            'BODY_HEIGHT': sel_height,
            'HR_PULSE': sel_pulse,
            'BP_HIGH': sel_bphigh,
            'BP_LOW': sel_bplow,
            'RR': sel_rr,
            'HP': sel_HB,
            'UREA': sel_urea,
            'CREATININIE': sel_cre,
            'MODE_OF_ARRIVAL': arr,
            'STATE': state,
            'admission': adm,
            'total_stay': sel_stay,
            'ICU': sel_icu,
            'Ward': sel_ward,
            'imp': sel_imp,
            'dia1': dia1,
            'dia2': dia2,
            'hyp1': hyp1,
            'hyp2': hyp2,
            'hyp3': hyp3,
            'other': oth,
            'ach': ach,
            'cdvd': cdvd,
            'csvd': csvd,
            'ctvd': ctvd,
            'cvsd': cvsd,
            'oasd': oasd,
            'pvsd': pvsd,
            'rhd': rhd,
            'heart': heart,
            'resp': resp,
            'gen': gen,
            'ner': ner,
            'ter': ter            
            }

inp = pd.DataFrame(data, index=[0])

# Slider
with st.form('Form1'):
    col2.header('Model Performance ')
    col3.header(' ')
    col3.header(' ')
    
    col2.subheader('Train Test Split')
#sel_alg = col2.selectbox(' List of Applicable Algorithms', ('LinearRegression', 'Lasso', 'Ridge', 'ElasticNet', 'Support Vector', 'Decision Tree', 'Naive Bias', 'Random Forest', 'XGBoost', 'Light Gradient Boost'))
    sel_split = col2.slider('Choose Train Test Split Percentage', 0.1, 0.9, 0.2, 0.01)
    col3.subheader('Random State')
    sel_random = col3.slider('Choose Random State Value', 1, 100, 35, 1)
    submitted = st.form_submit_button('Submit')


# Data Preprocessing

df = pd.read_csv('data.csv')
df.AGE = df.AGE.astype(int)
df['GENDER'] = df['GENDER'].replace({'M': 1, 'F': 0})
df['MARITAL STATUS'] = df['MARITAL STATUS'].replace({'UNMARRIED': 0, 'MARRIED': 1})
df['PAST MEDICAL HISTORY CODE'] = df['PAST MEDICAL HISTORY CODE'].replace({'Hypertension1': 'hypertension1'})
gd = pd.get_dummies(df['PAST MEDICAL HISTORY CODE'])
df1 = pd.concat([df, gd],axis =1)
df1['MODE OF ARRIVAL'] = df1['MODE OF ARRIVAL'].replace({'WALKED IN': 1, 'AMBULANCE':2, 'TRANSFERRED': 3})
df1['STATE AT THE TIME OF ARRIVAL'] = np.where(df1['STATE AT THE TIME OF ARRIVAL'] == 'ALERT', 1, 0)
df1['TYPE OF ADMSN'] = np.where(df1['TYPE OF ADMSN'] == 'ELECTIVE', 0, 1)
df1['IMPLANT USED (Y/N)'] = np.where(df1['IMPLANT USED (Y/N)'] == 'Y', 1, 0)
get = pd.get_dummies(df1['KEY COMPLAINTS -CODE'])
df2 = pd.concat([df1, get], axis=1)
df2.columns = df2.columns.str.replace(' ','_')
df2.columns = df2.columns.str.replace('-','_')
df2['BP__HIGH'] = df2.groupby(['Diabetes1', 'Diabetes2',  'hypertension1', 'hypertension2', 'hypertension3']).BP__HIGH.transform(lambda x: x.fillna(x.mean()))
df2['BP_LOW'] = df2.groupby(['Diabetes1', 'Diabetes2',  'hypertension1', 'hypertension2', 'hypertension3']).BP_LOW.transform(lambda x: x.fillna(x.mean()))
del df2['PAST_MEDICAL_HISTORY_CODE'], df2['KEY_COMPLAINTS__CODE']
df2['HB']  = df2['HB'].transform( lambda x: x.fillna(x.mean()))
df2['UREA'] = df2.groupby(['Diabetes1', 'Diabetes2',  'hypertension1', 'hypertension2', 'hypertension3']).UREA.transform(lambda x: x.fillna(x.mean()))
df2['CREATININE'] = df2.groupby(['Diabetes1', 'Diabetes2', 'hypertension1', 'hypertension2', 'hypertension3']).CREATININE.transform(lambda x: x.fillna(x.mean()))
y = df2.pop('TOTAL_COST_TO_HOSPITAL_')
s = df2.pop('SL.')
i = df2.pop('IMPLANT_USED_(Y/N)')
X = df2


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = sel_split, random_state = sel_random )



lm = LinearRegression()
lm.fit(X_train, y_train)
pred_train = lm.predict(X_train)
pred_test = lm.predict(X_test)
pred_test = abs(pred_test)
lmr2scoretrain = r2_score(y_train, pred_train)
lmr2scoretest = r2_score(y_test, pred_test)
st.write('### Algorithm Performance Comparision')
col4, col5 = st.beta_columns((1,1))
col4.subheader('Linear Reg  R2 Score(Train) ')
col4.write(lmr2scoretrain)
col5.subheader('Linear Reg  R2 Score(Test) ')
col5.write(lmr2scoretest)


ls = Lasso()
ls.fit(X_train, y_train)
lspred_train = ls.predict(X_train)
lspred_test = ls.predict(X_test)
lspred_test = abs(lspred_test)
lsr2scoretrain = r2_score(y_train, lspred_train)
lsr2scoretest = r2_score(y_test, lspred_test)
col4.subheader('Lasso R2 Score(Train) ')
col4.write(lsr2scoretrain)
col5.subheader('Lasso R2 Score(Test) ')
col5.write(lsr2scoretest)


ri = Ridge()
ri.fit(X_train, y_train)
ripred_train = ri.predict(X_train)
ripred_test = ri.predict(X_test)
ripred_test = abs(ripred_test)
rir2scoretrain = r2_score(y_train, ripred_train)
rir2scoretest = r2_score(y_test, ripred_test)
col4.subheader('Ridge R2 Score(Train) ')
col4.write(rir2scoretrain)
col5.subheader('Ridge R2 Score(Test) ')
col5.write(rir2scoretest)

el = ElasticNet()
el.fit(X_train, y_train)
elpred_train = el.predict(X_train)
elpred_test = el.predict(X_test)
elpred_test = abs(elpred_test)
elr2scoretrain = r2_score(y_train, elpred_train)
elr2scoretest = r2_score(y_test, elpred_test)
col4.subheader('ElasticNet R2 Score(Train) ')
col4.write(elr2scoretrain)
col5.subheader('ElasticNet R2 Score(Test) ')
col5.write(elr2scoretest)

kn = KNeighborsRegressor(n_neighbors = 3, metric = 'manhattan', weights = 'distance', n_jobs = -1)
kn.fit(X_train, y_train)
knpred_train = kn.predict(X_train)
knpred_test = kn.predict(X_test)
knpred_test = abs(knpred_test)
knr2scoretrain = r2_score(y_train, knpred_train)
knr2scoretest = r2_score(y_test, knpred_test)
col4.subheader('KNN R2 Score(Train) ')
col4.write(knr2scoretrain)
col5.subheader('KNN R2 Score(Test) ')
col5.write(knr2scoretest)

rf = RandomForestRegressor(n_estimators = 12,
                                         min_samples_leaf= 3,
                                         max_features = 18,
                                         random_state = 100)
rf.fit(X_train, y_train)
rfpred_train = rf.predict(X_train)
rfpred_test = rf.predict(X_test)
rfpred_test = abs(rfpred_test)
rfr2scoretrain = r2_score(y_train, rfpred_train)
rfr2scoretest = r2_score(y_test, rfpred_test)
col4.subheader('Random Forest R2 Score(Train) ')
col4.write(rfr2scoretrain)
col5.subheader('Random Forest R2 Score(Test) ')
col5.write(rfr2scoretest)


xg = XGBRegressor()
xg.fit(X_train, y_train)
xgpred_train = xg.predict(X_train)
xgpred_test = xg.predict(X_test)
xgpred_test = abs(xgpred_test)
xgr2scoretrain = r2_score(y_train, xgpred_train)
xgr2scoretest = r2_score(y_test, xgpred_test)
col4.subheader('XGBoost R2 Score(Train) ')
col4.write(xgr2scoretrain)
col5.subheader('XGBoost R2 Score(Test) ')
col5.write(xgr2scoretest)

tr = DecisionTreeRegressor(min_samples_leaf= 3,
                                         max_features = 18,
                                         random_state = 100)
tr.fit(X_train, y_train)
trpred_train = tr.predict(X_train)
trpred_test = tr.predict(X_test)
trpred_test = abs(trpred_test)
trr2scoretrain = r2_score(y_train, trpred_train)
trr2scoretest = r2_score(y_test, trpred_test)
col4.subheader('Decision Tree R2 Score(Train) ')
col4.write(trr2scoretrain)
col5.subheader('Decision Tree R2 Score(Test) ')
col5.write(trr2scoretest)



vt = VotingRegressor([('rf',rf), ('tr',tr), ('xg',xg), ('kn',kn) ])
vt.fit(X_train, y_train)
vtpred_train = vt.predict(X_train)
vtpred_test = vt.predict(X_test)
vtpred_test = abs(vtpred_test)
vtr2scoretrain = r2_score(y_train, vtpred_train)
vtr2scoretest = r2_score(y_test, vtpred_test)
col4.subheader('Voting Regressor R2 Score(Train) ')
col4.write(vtr2scoretrain)
col5.subheader('Voting Regressor R2 Score(Test) ')
col5.write(vtr2scoretest)

st = StackingRegressor([('lm',lm), ('ri',ri), ('el',el), ('ls',ls), ('rf',rf) ] , final_estimator=LinearRegression())
st.fit(X_train, y_train)
stpred_train = st.predict(X_train)
stpred_test = st.predict(X_test)
stpred_test = abs(stpred_test)
str2scoretrain = r2_score(y_train, stpred_train)
str2scoretest = r2_score(y_test, stpred_test)
col4.subheader('Stacking R2 Score(Train) ')
col4.write(str2scoretrain)
col5.subheader('Stacking R2 Score(Test) ')
col5.write(str2scoretest)




if sel_alg ==  'LinearRegression':
   predcost = lm.predict(inp) 
elif sel_alg ==  'Lasso':
   predcost = ls.predict(inp) 
elif sel_alg ==  'Ridge':
   predcost = ri.predict(inp) 
elif sel_alg ==  'ElasticNet':
   predcost = el.predict(inp) 
elif sel_alg ==  'KNearest Neighbors':
   predcost = kn.predict(inp) 
elif sel_alg ==  'RandomForest':
   predcost = rf.predict(inp) 
elif sel_alg ==  'XGBoost':
   predcost = xg.predict(inp) 
elif sel_alg ==  'DecisionTree':
   predcost = tr.predict(inp) 
elif sel_alg ==  'Voting Regressor':
   predcost = vt.predict(inp) 
else:  
   predcost = st.predict(inp) 

cost = round(predcost[0], 2) 

col1.subheader('COST PREDICTION')
col1.markdown('<font color="blue">As per given Data & Preference, Approx cost to Hospital in INR</font>', unsafe_allow_html=True)
col1.write (cost)























