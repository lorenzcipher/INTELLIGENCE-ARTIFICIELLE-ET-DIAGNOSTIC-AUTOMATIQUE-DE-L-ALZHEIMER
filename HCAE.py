import numpy as np
from sklearn import model_selection
import settings
import tensorflow as tf
from sklearn import svm
from sklearn.model_selection import train_test_split
from encoder import Encoder
from hypergraph_utils import construct_H_with_KNN, generate_G_from_H
import tensorflow.compat.v1 as tf
import streamlit as st
import pandas as pd
from sklearn import metrics
from sklearn.metrics import auc, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve,accuracy_score
from sklearn.model_selection import validation_curve
from matplotlib import pyplot as plt
from sklearn import preprocessing
flags = tf.app.flags
FLAGS = flags.FLAGS

from scipy.interpolate import make_interp_spline, BSpline
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.svm import SVC
from sklearn.metrics import mean_squared_error,mean_absolute_error









"""Main function of HyperConnectome AutoEncoder(HCAE) for training and predicting
    HCAE(samples, labels, view)
    Inputs:
        view: An integer higher or equal to zero that selects what single view will the data samples be taken from
              -1 for fusing all the views together
        labels: (n x 1) matrix containing the labels of the data samples [0, 1]
                 n is the number of samples in the dataset
        samples: if view is different from -1 (multi-view):
			(n x m x m) matrix of original data
			n is the number of samples in dataset
                        m is the number of nodes
		if multi-view:
			(l x n x m x m) original data
			l is the number of views (always 4 for our data)
                        n is the number of samples in dataset
                        m is the number of nodes

    Outputs:
        result: (n x m x 1) matrix containing the extracted embeddings
                n the number of samples in the dataset
                m the number of nodes
        mean_err: the mean of classification error over the samples of the test set
        std_err:  the standard deviation of classification error over the samples of the test set
        avg_cost: mean of reconstruction error of the samples of the dataset
        
    Feel free to modify the hyperparameters in the settings.py file according to your needs"""


def HCAE(samples, labels, view=0):
    model = 'hyper_arga_ae'
    settings1 = settings.get_settings_new(model, view)

    enc = Encoder(settings1)
    if view == -1:
        adjacency0 = samples[0]
        adjacency1 = samples[1]
        adjacency2 = samples[2]
        adjacency3 = samples[3]
    else:
        adjacency0 = samples

    
    subject = adjacency0.shape[0]
    
    n = adjacency0.shape[1]

    result = np.zeros((subject, n, 1))

    H = []
    G = []

    for i in range(0, subject):
        if view == -1:
            H.append(np.concatenate((adjacency0[i], adjacency1[i], adjacency2[i], adjacency3[i]), axis=1))
        else:
            H.append(adjacency0[i])

    hypergraphs = []
    for i in range(subject):
        if view == -1:
            hypergraph0 = construct_H_with_KNN(adjacency0[i], K_neigs=FLAGS.multi_view_K)  # optimal value for multi-view is 13
            hypergraph1 = construct_H_with_KNN(adjacency1[i], K_neigs=FLAGS.multi_view_K)
            hypergraph2 = construct_H_with_KNN(adjacency2[i], K_neigs=FLAGS.multi_view_K)
            hypergraph3 = construct_H_with_KNN(adjacency3[i], K_neigs=FLAGS.multi_view_K)
            hypergraph = np.concatenate((hypergraph0, hypergraph1, hypergraph2, hypergraph3), axis=1)
        else:
            hypergraph = construct_H_with_KNN(H[i], K_neigs=FLAGS.single_view_K)  # optimal values for our single views 7,8,11,9,13
        hypergraphs.append(hypergraph)
    
    
    hypergraphs = np.asarray(hypergraphs)
    
    for i in range(0, subject):
        G.append(generate_G_from_H(hypergraphs[i]))
    
    

    
    
    G = np.asarray(G)

    print("G",np.shape(G))
    print("H",np.shape(H))

    for i in range(0, subject):
        print(' ')
        print('Subject: ' + str(i + 1))
        encoded_view,train_loss = enc.erun(H[i], G[i])
        result[i] = encoded_view
    

    
    
    

    avg_cost = enc.cost / subject
    print('Avg cost: ' + str(avg_cost))

    X = np.zeros((subject, n))

    for i in range(result.shape[0]):
        X[i] = result[i].transpose()

    print("Result",np.shape(X))
    

    
    
    
    #st.line_chart(train_loss)


    

    ############################### SVM ###################################
    
   
    
    accuracies = []
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, shuffle=True)
    
    # standardiser les données
    std_scale = preprocessing.StandardScaler().fit(X_train)

    X_train_std = std_scale.transform(X_train)
    X_test_std = std_scale.transform(X_test)

    # choisir 6 valeurs pour C, entre 1e-2 et 1e3
    C_range = np.logspace(-2, 5, 50)

    # choisir 4 valeurs pour gamma, entre 1e-2 et 10
    gamma_range = np.logspace(-2, 5, 50)

    # grille de paramètres
    param_grid = {'C': C_range, 'gamma': gamma_range}

    # critère de sélection du meilleur modèle
    score = 'roc_auc'

    # initialiser une recherche sur grille
    grid = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), 
                                        param_grid, 
                                        cv=5, # 5 folds de validation croisée  
                                        scoring=score)

    # faire tourner la recherche sur grille
    grid.fit(X_train_std, y_train)

    # afficher les paramètres optimaux
    print("The optimal parameters are {} with a score of {:.2f}".format(grid.best_params_, grid.best_score_))
        
        
        
    # prédire sur le jeu de test avec le modèle optimisé
    y_test_pred_cv = grid.predict(X_test_std)
    #grid.decision_function(X_test_std)
    

    # construire la courbe ROC du modèle optimisé
    fpr_cv, tpr_cv, thr_cv = metrics.roc_curve(y_test, y_test_pred_cv)

    # calculer l'aire sous la courbe ROC du modèle optimisé
    auc_cv = metrics.auc(fpr_cv, tpr_cv)

        
        
    

    
    
    
    
    
    

    ##################### Layout Application ##################
    container1 = st.container()
    col1, col2 = st.columns(2)

    with container1 :
        with col1:
            
        
            # créer une figure
            fig = plt.figure(figsize=(6, 6))
            


            # afficher la courbe ROC du modèle optimisé
            plt.plot(fpr_cv, tpr_cv, "--", lw=2, label='ROC curve (area = %0.2f)' % 
                    auc_cv,)
                    

            # donner un titre aux axes et au graphique
            plt.xlabel('False Positive Rate', fontsize=16)
            plt.ylabel('True Positive Rate', fontsize=16)
            plt.title('SVM ROC Curve', fontsize=16)

            # afficher la légende
            plt.legend(loc="lower right", fontsize=14)



            st.pyplot(fig)



        with col2:
            
            
            fig2 = plt.figure(figsize=(6, 6))
            param_range = np.logspace(-6, -1, 5)
            train_scores, test_scores = validation_curve(
                svm.SVC(),
                X_test,
                y_test,
                param_name="gamma",
                param_range=param_range,
                scoring="accuracy",
                n_jobs=-1,
            )
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            
            plt.title("Validation Curve with SVM")
            plt.xlabel(r"$\gamma$")
            plt.ylabel("Score")
            plt.ylim(0.0, 1.1)
            lw = 2
            plt.semilogx(
                param_range, train_scores_mean, label="Training score", color="darkorange", lw=lw
            )
            plt.fill_between(
                param_range,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.2,
                color="darkorange",
                lw=lw,
            )
            plt.semilogx(
                param_range, test_scores_mean, label="Cross-validation score", color="navy", lw=lw
            )
            plt.fill_between(
                param_range,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.2,
                color="navy",
                lw=lw,
            )
            plt.legend(loc="best")

            st.pyplot(fig2)

    container2 = st.container()
    with container2:  
        
        st.markdown('Loss Metrics **_MAE,MSE,RMSE_**.')
        fig3 = plt.figure(figsize=(6, 6))

        

        def create_spline_from(x, y, resolution):
            new_x = np.linspace(x[0], x[-1], resolution)
            y_spline = make_interp_spline(x, y, k=3)
            new_y= y_spline(new_x)

            return (new_x, new_y)

        epochs=list(range(5000,50001,5000))
        print(epochs)
        
        
        mae_loss=[0.500225365,0.000221096,0.000060971,0.000060323,0.000059905,0.000059579,0.000059274,0.000058972,0.000058697,0.000058476]

        
        
        
        
        
        mse_loss=[0.135419831,0.018331185,0.002481434,0.000335913,0.000045486,0.000006180,0.000000867,0.000000147,0.000000042,0.000000042]
        
        rmse_loss=[0.500225306,0.000293739,0.000126985,0.000121944,0.000119484,0.000117791,0.000116400,0.000115198,0.000114148,0.000113228]

        plt.semilogy(epochs, mae_loss, 'b', label='MAE')
        plt.semilogy(epochs, mse_loss, 'r', label='MSE')
        plt.semilogy(epochs, rmse_loss, 'g', label='RMSE')
        plt.legend()
        plt.show()
                

        st.pyplot(fig3)
    '''
    result = 0
    y_test = y_test.ravel()
    t = t.transpose()
    for i in range(y_test.shape[0]):
        result += (y_test[i] == t[i])

    accuracy = float(result) / y_test.shape[0]
    accuracies.append(accuracy)

    mean_err = np.mean(np.asarray(accuracies))
    std_err = np.std(np.asarray(accuracies))
    print('------------------')
    print('HCAE')
    print('Mean: ' + str(mean_err * 100) + '%')
    print('Std: ' + str(std_err * 100) + '%')
    print(' ')
    print("resultat ",result)

    # choisir 6 valeurs pour C, entre 1e-2 et 1e3
    C_range = np.logspace(-2, 3, 6)

    # choisir 4 valeurs pour gamma, entre 1e-2 et 10
    gamma_range = np.logspace(-2, 1, 4)

    # grille de paramètres
    param_grid = {'C': C_range, 'gamma': gamma_range}

    # critère de sélection du meilleur modèle
    score = 'roc_auc'

    # initialiser une recherche sur grille
    grid = model_selection.GridSearchCV(svm.SVC(kernel='rbf'), 
                                        param_grid, 
                                        cv=5, # 5 folds de validation croisée  
                                        scoring=score)

    # faire tourner la recherche sur grille
    grid.fit(X_train, y_train)

    # afficher les paramètres optimaux
    print("The optimal parameters are {} with a score of {:.2f}".format(grid.best_params_, grid.best_score_))
    from matplotlib import pyplot as plt
    # prédire sur le jeu de test avec le modèle optimisé
    y_test_pred_cv = grid.decision_function(X_test)

    # construire la courbe ROC du modèle optimisé
    fpr_cv, tpr_cv, thr_cv = metrics.roc_curve(y_test, y_test_pred_cv)

    # calculer l'aire sous la courbe ROC du modèle optimisé
    auc_cv = metrics.auc(fpr_cv, tpr_cv)

    # créer une figure
    fig = plt.figure(figsize=(6, 6))



    # afficher la courbe ROC du modèle optimisé
    plt.plot(fpr_cv, tpr_cv, '-', lw=2, label='gamma=%.1e, AUC=%.2f' % \
            (grid.best_params_['gamma'], auc_cv))
            

    # donner un titre aux axes et au graphique
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('SVM ROC Curve', fontsize=16)

    # afficher la légende
    plt.legend(loc="lower right", fontsize=14)

    # afficher l'image
    plt.show()
    '''
    '''
    chart_data = pd.DataFrame(
    accuracies,
    columns=["accuracies"])

    st.line_chart(chart_data)

    st.write(mean_err)
    '''
    #return result, mean_err, std_err, avg_cost

