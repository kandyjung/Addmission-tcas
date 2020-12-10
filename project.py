import streamlit as st
import pandas as pd
import pickle 

st.write(""" 

## My First Web Application 
Let's enjoy **data science** project! 

""")

st.sidebar.header('User Input') 
st.sidebar.subheader('Please enter your data:')

# -- Define function to display widgets and store data
def get_input():
    # Display widgets and store their values in variables
    v_Sex = st.sidebar.radio('Sex', ['Male','Female'])
    v_EntryTypeID = st.sidebar.slider('EntryTypeID', 10, 69, 30)
    v_TCAS = st.sidebar.radio('TCAS', ['1','2','3','4','5'])
    v_AcademicYear = st.sidebar.selectbox('AcademicYear', ['2562','2563'])
    v_PrefixName = st.sidebar.radio('PrefixName', ['MR.','MISS'])
    v_AcademicSemester = st.sidebar.selectbox('AcademicSemester', ['1'])
    v_FacultyName = st.sidebar.radio('FacultyName', 
    ['School of Liberal Arts','School of Management','School of Information Technology','School of Cosmetic Science','School of Medicine',
    'School of Law','School of Health Science','School of Agro-industry','School of Medicine','School of Sinology','School of Dentistry',
    'School of Social Innovation','School of Integrative Medicine','School of Nursing'
    ])
    v_LevelID = st.sidebar.selectbox('LevelID', ['3'])
    v_LevelNameEng = st.sidebar.selectbox('LevelNameEng', ['Undergraduate'])
    v_HomeRegion = st.sidebar.selectbox('HomeRegion', ['International','North','North East','South','Central','Bankok','East'])
    v_StudentTH = st.sidebar.radio('StudentTH',['FOREIGN','THAI'])
    v_SchoolRegionNameEng =  st.sidebar.selectbox('SchoolRegionNameEng',['Foreign','Northern','Central','Southern','Eastern'])
    v_Status = st.sidebar.radio('Status',['MFU student','Not MFU student'])

    # Change the value of sex to be {'M', 'F', 'I'} as stored in the trained dataset
    if v_StudentTH == 'FOREIGN':
        v_StudentTH = 1
    else :
        v_StudentTH = 0
    
    if v_Sex == 'Male':
        v_Sex = 'M'
    else :
        v_Sex = 'F'
    
    if v_PrefixName == 'MR.':
        v_PrefixName = 1
    else :
        v_PrefixName = 0

    if v_FacultyName == 'School of Liberal Arts':
        v_FacultyName = 1
    elif v_FacultyName == 'School of Management':
        v_FacultyName = 2
    elif v_FacultyName == 'School of Information Technology':
        v_FacultyName = 3
    elif v_FacultyName == 'School of Cosmetic Science':
        v_FacultyName = 4
    elif v_FacultyName == 'School of Medicine':
        v_FacultyName = 5
    elif v_FacultyName == 'School of Law':
        v_FacultyName = 6
    elif v_FacultyName == 'School of Health Science':
        v_FacultyName = 7
    elif v_FacultyName == 'School of Agro-industry':
        v_FacultyName = 8
    elif v_FacultyName == 'School of Medicine':
        v_FacultyName = 9
    elif v_FacultyName == 'School of Sinology':
        v_FacultyName = 10
    elif v_FacultyName == 'School of Dentistry':
        v_FacultyName = 11
    elif v_FacultyName == 'School of Social Innovation':
        v_FacultyName = 12
    elif v_FacultyName == 'School of Integrative Medicine':
        v_FacultyName = 13
    elif v_FacultyName == 'School of Nursing':
        v_FacultyName = 14
    else :
        v_PrefixName = 0

    if v_LevelNameEng == 'Undergraduate':
        v_LevelNameEng = 1
    else :
        v_PrefixName = 0

    if v_HomeRegion == 'International':
        v_HomeRegion = 1
    elif v_HomeRegion == 'North':
        v_HomeRegion = 2
    elif v_HomeRegion =='North East':
        v_HomeRegion = 3
    elif v_HomeRegion =='South':
        v_HomeRegion = 4
    elif v_HomeRegion =='Central':
        v_HomeRegion = 5
    elif v_HomeRegion =='Bankok':
        v_HomeRegion = 6
    elif v_HomeRegion =='East':
        v_HomeRegion = 7
    else :
        v_HomeRegion = 0


    if v_SchoolRegionNameEng == 'Foreign':
        v_SchoolRegionNameEng = 1
    elif v_SchoolRegionNameEng =='Northern':
        v_SchoolRegionNameEng = 2
    elif v_SchoolRegionNameEng =='Central':
        v_SchoolRegionNameEng = 3
    elif v_SchoolRegionNameEng =='Southern':
        v_SchoolRegionNameEng = 4
    elif v_SchoolRegionNameEng =='Eastern':
        v_SchoolRegionNameEng = 5
    else :
        v_SchoolRegionNameEng = 0

    if v_Status  == 'MFU student':
       v_Status  = 1
    else :
       v_Status  = 0


    # Store user input data in a dictionary
    data = {'Sex': v_Sex,
            'EntryTypeID': v_EntryTypeID,
            'TCAS': v_TCAS,
            'PrefixName': v_PrefixName,
            'AcademicSemester': v_AcademicSemester,
            'AcademicYear': v_AcademicYear,
            'FacultyName' : v_FacultyName,
            'LevelID' : v_LevelID,
            'LevelNameEng' : v_LevelNameEng,
            'HomeRegion': v_HomeRegion,
            'StudentTH': v_StudentTH,
            'SchoolRegionNameEng' : v_SchoolRegionNameEng,
            'Status' : v_Status
    }

    # Create a data frame from the above dictionary
    data_df = pd.DataFrame(data, index=[0])
    return data_df

# -- Call function to display widgets and get data from user
df = get_input()

st.header('Application of Abalone\'s Age Prediction:')

# -- Display new data from user inputs:
st.subheader('User Input:')
st.write(df)

# -- Data Pre-processing for New Data:
# Combines user input data with sample dataset
# The sample data contains unique values for each nominal features
# This will be used for the One-hot encoding
data_sample = pd.read_csv('newtcas2.csv')
df = pd.concat([df, data_sample],axis=0)

#One-hot encoding for nominal features
cat_data = pd.get_dummies(df[['HomeRegion']])

#Combine all transformed features together
X = pd.concat([cat_data, df], axis=1)
X = X[:1] # Select only the first row (the user input data)

#Drop un-used feature
X = X.drop(columns=['Sex','StudentType','EntryGroupName','Country','SchoolProvince'])

# -- Display pre-processed new data:
st.subheader('Pre-Processed Input:')
st.write(X)