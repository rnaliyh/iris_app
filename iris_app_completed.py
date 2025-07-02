import streamlit as st
import pickle
import time
import pandas as pd

st.set_page_config(page_title='ML Portfolio', page_icon='📊', layout='wide')

st.write('Welcome to the ML Portfolio App!')

select_var=st.sidebar.selectbox("Want to open about?",("Home","Iris Species","Heart Disease"))

# Define the iris function
def iris():
    st.write("""
    This app predicts the **Iris Species**
    
    Data obtained from the [iris dataset](https://www.kaggle.com/uciml/iris) by UCIML. 
    """)

    st.sidebar.header('User Input Features:')

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Input Manual')
            SepalLengthCm = st.sidebar.slider('Sepal Length (cm)', min_value=4.3, value=6.5, max_value=10.0)
            SepalWidthCm = st.sidebar.slider('Sepal Width (cm)', min_value=2.0, value=3.3, max_value=5.0)
            PetalLengthCm = st.sidebar.slider('Petal Length (cm)',min_value= 1.0, value=4.5, max_value=9.0)
            PetalWidthCm = st.sidebar.slider('Petal Width (cm)',min_value= 0.1, value=1.4, max_value=5.0)
            data = {'SepalLengthCm': SepalLengthCm,
                    'SepalWidthCm': SepalWidthCm,
                    'PetalLengthCm': PetalLengthCm,
                    'PetalWidthCm': PetalWidthCm}
            features = pd.DataFrame(data, index=[0]) #transform to pandas
            return features
        input_df = user_input_features()
    # # img = Image.open("iris.JPG")

    st.image("https://www.easytogrowbulbs.com/cdn/shop/products/BeardedIrisColorfullMix_VIS-sqWeb_8a293612-7bc0-4a9f-89ac-917e820d0ccb.jpg?v=1664472481&width=1920")

    button_var = st.sidebar.button('Predict!')

    if button_var:
        df = input_df
        st.write(df)
        
        with open("generate_iris.pkl", 'rb') as file:  
            loaded_model = pickle.load(file) # load previous trained model
        prediction = loaded_model.predict(df) # do prediction according to input value
        result = ['Iris-setosa' if prediction == 0 else ('Iris-versicolor' if prediction == 1 else 'Iris-virginica')]
        
        st.subheader('Prediction: ')
        output = str(result[0])
        with st.spinner('Wait for it...'):
            time.sleep(4)
            st.success(f"Prediction of this app is {output}")

# Define the heart function
def heart():
    st.write("""
    This app predicts the **Heart Disease Risk**
    
    Data obtained from the [Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
    
    """)
    st.sidebar.header('User Input Features:')

    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', min_value=0, value=1, max_value=3, step=1, help="Type of chest pain experienced")
            if cp == 0:
                wcp = "Typical Angina"
            elif cp == 1:
                wcp = "Atypical Angina"
            elif cp == 2:
                wcp = "Non-Anginal Pain"
            else:
                wcp = "Asymptomatic"
            st.sidebar.write(f"Chest Pain Type: {wcp}")

            thalach = st.sidebar.slider('Maximum Heart Rate Achieved', min_value=60, value=150, max_value=220, step=1, help="Maximum heart rate achieved during exercise")
            slope = st.sidebar.selectbox('Slope of ST Segment', options=[0, 1, 2], index=0, help="Slope of the peak exercise ST segment")
            oldpeak = st.sidebar.slider('Oldpeak', min_value=0.0, value=1.0, max_value=6.2, step=0.1, help="ST depression induced by exercise relative to rest")
            exang = st.sidebar.radio('Exercise Induced Angina', options=['Yes', 'No'], index=0, help="Whether exercise induced angina is present")
            if exang == 'Yes':
                exang = 1
            else:
                exang = 0
            ca = st.sidebar.selectbox('Number of Major Vessels', options=[0, 1, 2, 3], index=0, help="Number of major vessels colored by fluoroscopy")
            thal = st.sidebar.selectbox('Thalassemia', options=[1, 2, 3], index=0, help="Thalassemia result")
            sex= st.sidebar.radio('Sex', options=['Male', 'Female'], index=0)
            if sex=="Female":
                sex=0
            else :
                sex=1
            age = st.sidebar.number_input('Age', min_value=29, max_value=77, value=30, step=1, help="Age of the patient in years")

            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex' : sex,
                    'age': age}
            
            # Create a DataFrame from the input data
            features = pd.DataFrame(data, index=[0])
            return features

        input_df=user_input_features()
        st.image("https://drramjimehrotra.com/wp-content/uploads/2022/09/Women-Heart-Disease-min-resize.png")

        if st.sidebar.button('Predict!'):
            df= input_df
            st.write(df)
            with open("generate_heart_disease.pkl", 'rb') as file:  
                loaded_model = pickle.load(file)

            prediction_proba = loaded_model.predict_proba(df)
            if prediction_proba[:,1] >= 0.4:
                prediction = 1
            else: 
                prediction = 0
            
            result = ['No Heart Disease Risk' if prediction == 0 else 'Heart Disease Risk Detected']
            
            # Print the prediction result
            st.subheader('Prediction: ')
            output = str(result[0])
            with st.spinner('Wait for it...'):
                time.sleep(4)
                if output == "No Heart Disease Risk":
                    st.success(f"Prediction : {output}")
                if output == "Heart Disease Risk Detected":
                    st.error(f"Prediction : {output}")
                    st.info("Please consult a doctor for further evaluation and advice.")

if select_var == "Iris Species":
    iris()
elif select_var == "Heart Disease":
    heart()


