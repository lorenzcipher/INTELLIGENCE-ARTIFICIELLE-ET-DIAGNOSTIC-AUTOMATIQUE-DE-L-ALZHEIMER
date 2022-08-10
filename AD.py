# coding=utf-8
import os 
from doctest import master
import streamlit as st

from HCAE import HCAE
from simulate_data import simulate_data, simulate_labels
import pandas as pd
import nibabel as nib

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.naive_bayes import GaussianNB

# Packages
from streamlit import caching
import time
import matplotlib.pyplot as plt

from simulate_data import simulate_data, simulate_labels
from sklearn import datasets
from streamlit_option_menu import option_menu
import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np



import plotly.express as px
import io 
import warnings
warnings.filterwarnings("ignore")
from HCAE import HCAE
from simulate_data import simulate_data, simulate_labels


def main():
    
    with st.sidebar:
        selected = option_menu("Menu", ["Home",'Visualization fMRI','Morphological Data','Data Analyse','Metrics'], 
            icons=['house','eye','kanban','sliders','activity'], default_index=0)

    
    logo = Image.open(r'Brain-Logo.png')
    profile = Image.open(r'image_bg.jpg')


    if selected == "Home":
        col1, col2 = st.columns( [0.8, 0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'BodoniFLF'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">About Our Project</p>', unsafe_allow_html=True)    
        with col2:               # To display brand log
            st.image(logo, width=130 )
        
        
        st.write('''Of all the medical data, images are the most important in terms of volume of data, information exploited and knowledge extracted. Medical imaging makes it possible to visualize the inside of a human (or animal) body without surgery. It is used for clinical purposes, i.e. in the search for a diagnosis or a treatment of certain pathologies, but also for scientific research in order to study the reactions of the body of the living being in front of diseases (Cancer, Autism, Alzheimer's, etc.) which have recently preoccupied doctors and researchers in the medical field.
        \n The systematic production of digitized medical data means that computer analysis and synthesis have developed very dramatically over the last twenty years, leading to the automation of the diagnosis of various diseases and numerous studies requiring multiple medical image processing. Thus, computer tools have become indispensable, even unavoidable, in the face of this data structure.
        \n Among the many challenges that imaging can exploit in medical research is Alzheimer's disease, the most common form of dementia.
        \n ''')
        st.image(profile, width=400) 
        st.write("Alzheimer's disease is a neurodegenerative brain disorder for which there is currently no cure. Early diagnosis of this disease by Deep Learning applied to functional magnetic resonance imaging (fMRI) of the brain and specifically by convolutional neural networks (CNN), is attracting increasing interest due to their high performance in classification, detection and segmentation in medical imaging.")   
        
    elif selected == "Visualization fMRI":
        col1, col2 = st.columns( [0.8, 0.2])
        with col1:               # To display the header text using css style
            st.markdown(""" <style> .font {
            font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
            </style> """, unsafe_allow_html=True)
            st.markdown('<p class="font">Upload fMRI data...</p>', unsafe_allow_html=True)
            
        with col2:               # To display brand logo
            st.image(logo,  width=150)
        #Add file uploader to allow users to upload photos
        uploaded_file = st.file_uploader("", type=['jpg','png','jpeg','nii'])
        if uploaded_file is not None:
            

            data = nib.load(uploaded_file.name).get_fdata()
            
            x = st.sidebar.slider('Axes X :', 0, data.shape[0]-1,data.shape[0]-1,5)
            y = st.sidebar.slider('Axes Y :', 0, data.shape[1]-1,50,5)
            z = st.sidebar.slider('Axes Z :', 0, data.shape[2]-1,data.shape[2]-1,5)
            container1 = st.container()
            col1, col2 ,col3 = st.columns(3)
            

            with container1:
                with col1:
                    fig = plt.figure()
                    plt.imshow(data[x,:,:])
                    plt.axis("off")
                    st.pyplot(fig)

                with col2 : 
                    fig = plt.figure()
                    plt.imshow(data[:,y,:])
                    plt.axis("off")
                    st.pyplot(fig)
            
                with col3:
                    plt.imshow(data[:,:,z])
                    plt.axis("off")
                    st.pyplot(fig)
            
            container2 = st.container()
            col4, col5 ,col6 = st.columns(3)
            
            mask = nib.load(uploaded_file.name).get_fdata()
            with container2:
                with col4:
                    fig_mask = plt.figure()
                    plt.imshow(data[x,:,:], 'gray', interpolation='none')
                    plt.imshow(mask[x,:,:], 'jet', alpha=0.5, interpolation='none')
                    plt.axis("off")
                    st.pyplot(fig_mask)

                with col5:
                    fig_mask = plt.figure()
                    plt.imshow(data[:,y,:], 'gray', interpolation='none')
                    plt.imshow(mask[:,y,:], 'jet', alpha=0.5, interpolation='none')
                    plt.axis("off")
                    st.pyplot(fig_mask)
                
                with col6:
                    fig_mask = plt.figure()
                    plt.imshow(data[:,:,z], 'gray', interpolation='none')
                    plt.imshow(mask[:,:,z], 'jet', alpha=0.5, interpolation='none')
                    plt.axis("off")
                    st.pyplot(fig_mask)


            

            


                        
            
            
             
    elif selected == "Morphological Data":
        #Add a file uploader to allow users to upload their project plan file
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Upload your project plan</p>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Fill out the project plan template and upload your file here. After you upload the file, you can edit your project plan within the app.", type=['csv'], key="2")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            to_drop = ['COLPROT',
            'VISCODE',
            'EXAMDATE',
            'VERSION',
            'update_stamp',
            'VISCODE2',
            'LONIUID',
            'IMAGEUID',
            'RUNDATE',
            'TEMPQC',
            'FRONTQC',
            'PARQC',
            'INSULAQC',
            'OCCQC',
            'BGQC',
            'CWMQC',
            'VENTQC',
            'HIPPOQC',
            'STATUS']


            data.drop(to_drop, inplace=True, axis=1)

            data.drop_duplicates(subset ="RID", keep = 'first', inplace=True)
            data = data.set_index('RID')

            colonnes = data.columns
            col_sv = [x for x in colonnes if x.endswith('SV')]
            data.drop(col_sv, inplace=True, axis=1)

            data['OVERALLQC'] = data['OVERALLQC'].replace(['Partial'],'AD')
            data['OVERALLQC'] = data['OVERALLQC'].replace(['Pass'],'MCI')
            index_with_nan = data.index[data.isnull().any(axis=1)]
            data.drop(index_with_nan, inplace=True)

            st.write(data)
        else:
            st.write('---') 

        
      
        

    elif selected == "Data Analyse":
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Alzheimer disease Dataset Analysis</p>', unsafe_allow_html=True)
        
        breast_cancer = datasets.load_breast_cancer(as_frame=True)
        breast_cancer_df = pd.concat((breast_cancer["data"], breast_cancer["target"]), axis=1)
        breast_cancer_df["target"] = [breast_cancer.target_names[val] for val in breast_cancer_df["target"]]
        uploaded_file = st.file_uploader("Fill out the project plan template and upload your file here. After you upload the file, you can edit your project plan within the app.", type=['csv'], key="2")
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            to_drop = ['COLPROT',
            'VISCODE',
            'EXAMDATE',
            'VERSION',
            'update_stamp',
            'VISCODE2',
            'LONIUID',
            'IMAGEUID',
            'RUNDATE',
            'TEMPQC',
            'FRONTQC',
            'PARQC',
            'INSULAQC',
            'OCCQC',
            'BGQC',
            'CWMQC',
            'VENTQC',
            'HIPPOQC',
            'STATUS']


            data.drop(to_drop, inplace=True, axis=1)

            data.drop_duplicates(subset ="RID", keep = 'first', inplace=True)
            data = data.set_index('RID')

            colonnes = data.columns
            col_sv = [x for x in colonnes if x.endswith('SV')]
            data.drop(col_sv, inplace=True, axis=1)

            data['OVERALLQC'] = data['OVERALLQC'].replace(['Partial'],'AD')
            data['OVERALLQC'] = data['OVERALLQC'].replace(['Pass'],'MCI')
            index_with_nan = data.index[data.isnull().any(axis=1)]
            data.drop(index_with_nan, inplace=True)



            ################# Scatter Chart Logic #################

            st.sidebar.markdown("### Scatter Chart: Explore Relationship Between Measurements :")

            measurements_0 = data.drop(labels=["OVERALLQC"], axis=1).columns.tolist()

            x_axis_0 = st.sidebar.selectbox("X-Axis", measurements_0)
            y_axis_0 = st.sidebar.selectbox("Y-Axis", measurements_0, index=1)

            scatter_fig_0 = plt.figure(figsize=(6,4))

            scatter_ax_0 = scatter_fig_0.add_subplot(111)

            malignant_df_0 = data[data["OVERALLQC"] == "AD"]
            benign_df_0 = data[data["OVERALLQC"] == "MCI"]

            malignant_df_0.plot.scatter(x=x_axis_0, y=y_axis_0, s=120, c="tomato", alpha=0.6, ax=scatter_ax_0, label="Pass")
            benign_df_0.plot.scatter(x=x_axis_0, y=y_axis_0, s=120, c="dodgerblue", alpha=0.6, ax=scatter_ax_0,
                                title="{} vs {}".format(x_axis_0.capitalize(), y_axis_0.capitalize()), label="Partial");


            

             ########## Bar Chart Logic ##################

            st.sidebar.markdown("### Bar Chart: Average ROI Measurements Per Alzheimer phase : ")

            avg_data_df = data.groupby("OVERALLQC").mean()
            bar_axis = st.sidebar.multiselect(label="Average ROI Measures per Alzheimer phase Type Bar Chart",
                                            options=measurements_0,
                                            default=["ST102TA","ST102CV", "ST102SA"])

            if bar_axis:
                bar_fig = plt.figure(figsize=(6,4))

                bar_ax = bar_fig.add_subplot(111)

                sub_avg_alzheimer_disease_df = avg_data_df[bar_axis]

                sub_avg_alzheimer_disease_df.plot.bar(alpha=0.8, ax=bar_ax, title="Average ROI Measurements per Alzheimer Phase");

            else:
                bar_fig = plt.figure(figsize=(6,4))

                bar_ax = bar_fig.add_subplot(111)

                sub_avg_breast_cancer_df = data[["ST102TA","ST102CV", "ST102SA"]]

                sub_avg_breast_cancer_df.plot.bar(alpha=0.8, ax=bar_ax, title="Average ROI Measurements per Alzheimer Phase");

            ################# Histogram Logic ########################

            st.sidebar.markdown("### Histogram: Explore Distribution of Measurements : ")

            hist_axis = st.sidebar.multiselect(label="Histogram Ingredient", options=measurements_0, default=["ST102TA","ST102CV"])
            bins = st.sidebar.radio(label="Bins :", options=[10,20,30,40,50], index=4)

            if hist_axis:
                hist_fig = plt.figure(figsize=(6,4))

                hist_ax = hist_fig.add_subplot(111)

                sub_alzheimer_disease_df = data[hist_axis]

                sub_alzheimer_disease_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");
            else:
                hist_fig = plt.figure(figsize=(6,4))

                hist_ax = hist_fig.add_subplot(111)

                sub_alzheimer_disease_df =data[["ST101SV","ST102CV"]]

                sub_alzheimer_disease_df.plot.hist(bins=bins, alpha=0.7, ax=hist_ax, title="Distribution of Measurements");
            
            
             #################### Hexbin Chart Logic ##################################

            st.sidebar.markdown("### Hexbin Chart: Explore Concentration of Measurements :")

            hexbin_x_axis = st.sidebar.selectbox("Hexbin-X-Axis", measurements_0, index=0)
            hexbin_y_axis = st.sidebar.selectbox("Hexbin-Y-Axis", measurements_0, index=1)

            if hexbin_x_axis and hexbin_y_axis:
                hexbin_fig = plt.figure(figsize=(6,4))

                hexbin_ax = hexbin_fig.add_subplot(111)

                data.plot.hexbin(x=hexbin_x_axis, y=hexbin_y_axis,
                                            reduce_C_function=np.mean,
                                            gridsize=25,
                                            #cmap="Greens",
                                            ax=hexbin_ax, title="Concentration of Measurements");
            
            

            ##################### Layout Application ##################
            container1 = st.container()
            col1, col2 = st.columns(2)

            with container1:
                with col1:
                    scatter_fig_0
                with col2:
                    bar_fig
            
            container2 = st.container()
            col3, col4 = st.columns(2)

            with container2:
                with col3:
                    hist_fig
                with col4:
                    hexbin_fig

    elif selected=='Metrics':
        
        st.sidebar.title("Parametres de Simulation")
        st.sidebar.subheader("Patients number")
        subjects = st.sidebar.slider('Nombre de patients', 0, 2500, 100,50)
        st.sidebar.subheader("number of MCI patients")
        mci = st.sidebar.slider('Nombre de patients malade de MCI', 0, 1250, 50,25)
        if  st.sidebar.button('Run'):
            if subjects and mci:
                view = -1
                labels = simulate_labels(mci, subjects)
                samples = []
                for i in range(4):
                    samples.append(simulate_data(35, subjects))

                HCAE(samples, labels, view) 
                
            else :
                st.sidebar.wrirte("Set the parameters")
                

        

    
  

    
        

        
       
           
        
   


if __name__ == "__main__":
    main()


