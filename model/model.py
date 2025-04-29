from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import plotly_express as px
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="Prediction Breast Cancer",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None)

alt.themes.enable("dark")
col,coli = st.columns([2,1])
with col:
    
    col1, col2= st.columns([0.5,3])
    with col1:
        st.image('/workspaces/bigdata/model/cancer.png')
    with col2:
        st.title(":blue[**Demo Classification du Cancer du Sein**]")
    
    tab1, tab2,tab3 = st.tabs(["### MODEL ET METRICS",
             "### Chart","### Prédictions"])
    
    with tab1:
        st.title(':orange[Chargement du dataset Breast Cancer Wisconsin]')
        
        data = pd.read_csv('/workspaces/bigdata/model/file_set.csv')
        data_set = data.drop('Unnamed: 0', axis=1)
        data_set
        data_set['diagnosis']=data_set['diagnosis'].map({'B':0,'M':1})
        model = joblib.load('/workspaces/bigdata/model/breast_cancer_model.pkl')
    
        X = data_set.drop('diagnosis', axis=1)
        y = data_set['diagnosis']
        std = StandardScaler()
        X_scalee = std.fit_transform(X)
        #prediction = model.predict(X_scalee)
        y_pred = model.predict(X_scalee)
    
        #accuracy du modèle
        st.write("### :blue[Accuracy du modèle]")
        model_acc = accuracy_score(y, y_pred)
        st.write(f":green[{model_acc}]")
    
            
        # Classification report
        st.write("### :blue[Report Classifier du modèle]")
        st.dataframe(pd.DataFrame(classification_report(y, y_pred, output_dict=True)).transpose())
        
         # Matrice de confusion
        st.write("### :blue[Matrice de Confusion]")
        
        
        
        cm = confusion_matrix(y, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benin', 'Malin'], yticklabels=['Bénin', 'Malin'])
        plt.xlabel("Prédictions")
        plt.ylabel("Données Réelles")
        st.pyplot(fig)
    
            
    with tab2:
        import streamlit as st
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        #predictions = ['maligne', 'bénigne', 'maligne', 'bénigne', 'bénigne', 'maligne']
        # 
        st.title(":orange[Distribution des prédictions de cancer]")
        fig, ax = plt.subplots()
        sns.countplot(x=y_pred, ax=ax)
        ax.set_xlabel("Type de cancer")
        ax.set_ylabel("Nombre de prédictions")
        st.pyplot(fig)
        ################
        #st.title("Histogramme des coefficients")
        # Obtenir les coefficients
    
        X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=0.2,
                    random_state=42)
        logreg = LogisticRegression()
        logreg.fit(X_train, y_train)
        predictions1 = logreg.predict(X_test)
        coefficients = logreg.coef_[0]
        
        # Calculer la valeur absolue des coefficients
        abs_coefficients = abs(coefficients)
        
        # Créer l'histogramme
        #plt.hist(abs_coefficients)
        #plt.xlabel("Valeur absolue des coefficients")
        #plt.ylabel("Nombre de variables")
        #plt.title("Histogramme des coefficients de la régression logistique")
        plt.show()



    
with tab3:
        # Configuration de l'interface Streamlit
    st.title(":orange[Démo de Classification du Cancer du Sein]")
    st.write(":blue[Utilisation d'un modèle de Machine Learning pour prédire si une tumeur est bénigne(Non Cancereuse) ou maligne(Cancereuse).]")
        # Saisie des caractéristiques
    features = X.columns.tolist()
    user_input = {feat: st.number_input(feat, min_value=0.01) for feat in features}
    if st.button("Prédire"):
        input_data = np.array([list(user_input.values())]).reshape(1, -1)
        x_scale = std.fit_transform(input_data)
        
        prediction = model.predict(x_scale)
        prediction_proba = model.predict_proba(x_scale)[0]
        
        st.write("### :green[Résultat de la Prédiction :]")
        st.write("🔴 Maligne(Tumeur Cancereuse)" if prediction == 1 else "🟢 Bénigne(Tumeur non cancereuse)")
        st.write(f"Probabilité : {prediction_proba[0]:.2%} d'être bénigne")
        st.write(f"Probabilité : {prediction_proba[1]:.2%} d'être maligne")
        st.success("Analyse terminée !")



    