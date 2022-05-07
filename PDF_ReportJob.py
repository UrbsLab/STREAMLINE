"""
File: PDF_ReportJob.py
Authors: Ryan J. Urbanowicz, Richard Zhang, Wilson Zhang
Institution: University of Pensylvania, Philadelphia PA
Creation Date: 6/1/2021
License: GPL 3.0
Description: Phase 8 of STREAMLINE - This 'Job' script is called by PDF_ReportMain.py which generates a formatted PDF summary report of key
pipeline results It is run once for the whole pipeline analysis.
"""
#Import required packages  ---------------------------------------------------------------------------------------------------------------------------
import pandas as pd
from fpdf import FPDF
from datetime import datetime
import os
import re
import sys
import pickle
import math
import glob

def job(experiment_path,training,rep_data_path,data_path):

    time = str(datetime.now())
    print(time)
    experiment_name = experiment_path.split('/')[-1]
    train_name = 'None'
    #Find folders inside directory
    if eval(training):
        ds = os.listdir(experiment_path)
        nonds = ['DatasetComparisons', 'jobs', 'jobsCompleted', 'logs','KeyFileCopy','metadata.pickle','metadata.csv','algInfo.pickle',experiment_name+'_ML_Pipeline_Report.pdf']
        for i in nonds:
            if i in ds:
                ds.remove(i)
        if '.idea' in ds:
            ds.remove('.idea')
        ds = sorted(ds)
    else:
        train_name = data_path.split('/')[-1].split('.')[0]
        ds = []
        for datasetFilename in glob.glob(rep_data_path+'/*'):
            datasetFilename = str(datasetFilename).replace('\\','/')
            apply_name = datasetFilename.split('/')[-1].split('.')[0] #Save unique dataset names so that analysis is run only once if there is both a .txt and .csv version of dataset with same name.
            ds.append(apply_name)
        ds = sorted(ds)

    #Unpickle metadata from previous phase
    file = open(experiment_path+'/'+"metadata.pickle", 'rb')
    metadata = pickle.load(file)
    file.close()

    #Unpickle algoritithm information from previous phase
    file = open(experiment_path+'/'+"algInfo.pickle", 'rb')
    algInfo = pickle.load(file)
    file.close()

    #Turn metadata dictionary into text list
    ars_dic = []
    for key in metadata:
        ars_dic.append(str(key)+':')
        ars_dic.append(str(metadata[key]))
        ars_dic.append('\n')

    #Analysis Settings, Global Analysis Settings, ML Modeling Algorithms
    analy_report = FPDF('P', 'mm', 'A4')
    analy_report.set_margins(left=10, top=5, right=10, )
    analy_report.add_page(orientation='P')
    top = analy_report.y

    #PDF page dimension reference - page width = 210 and page height down to start of footer = 285 (these are estimates)
    #FRONT PAGE - Summary of Pipeline settings-------------------------------------------------------------------------------------------------------
    print("Starting Report")
    ls1 = ars_dic[0:87] # Class - filter poor [0:87]
    ls2 = ars_dic[87:132]   # ML modeling algorithms (NaiveB - ExSTraCS) [87:132]
    ls3 = ars_dic[132:147]  # primary metric - hypersweep timeout [132:147]
    ls4 = ars_dic[147:162]  # LCS parameters (do LCS sweep - LCS hypersweep timeout) [147:162]
    ls5 = ars_dic[162:180]  # [162:180]
    analy_report.set_font('Times', 'B', 12)
    analy_report.cell(w=180, h=8, txt='STREAMLINE Training Summary Report: '+time, ln=2, border=1, align='L')
    analy_report.y += 2 #Margin below page header
    topOfList = analy_report.y #Page height for start of algorithm settings
    analy_report.set_font('Times', 'B', 10)
    analy_report.multi_cell(w = 90,h = 4,txt='General Pipeline Settings:', border=1, align='L')
    analy_report.y += 1 #Space below section header
    analy_report.set_font('Times','', 8)
    analy_report.multi_cell(w = 90,h = 4,txt=' '+listToString(ls1)+' '+listToString(ls3)+' '+listToString(ls5), border=1, align='L')
    bottomOfList = analy_report.y
    analy_report.x += 90
    analy_report.y = topOfList #96
    analy_report.set_font('Times', 'B', 10)
    analy_report.multi_cell(w = 90,h = 4,txt='ML Modeling Algorithms:', border=1, align='L')
    analy_report.y += 1 #Space below section header
    analy_report.set_font('Times','', 8)
    analy_report.x += 90
    analy_report.multi_cell(w = 90,h = 4,txt=' '+listToString(ls2), border=1, align='L')
    analy_report.x += 90
    analy_report.y += 2
    analy_report.set_font('Times', 'B', 10)
    analy_report.multi_cell(w = 90,h = 4,txt='LCS Settings (eLCS,XCS,ExSTraCS):', border=1, align='L')
    analy_report.y += 1 #Space below section header
    analy_report.set_font('Times','', 8)
    analy_report.x += 90
    analy_report.multi_cell(w = 90,h = 4,txt=' '+listToString(ls4), border=1, align='L')
    analy_report.y = bottomOfList + 2

    tryAgain = True
    try:
        analy_report.image('Pictures/STREAMLINE_LOGO.png', 102, 160, 90)
        tryAgain = False
    except:
        pass
    if tryAgain:
        try: #Running on Google Colab
            analy_report.image('/content/drive/MyDrive/STREAMLINE/Pictures/STREAMLINE_LOGO.png', 102, 160, 90)
        except:
            pass

    if eval(training):
        #Get names of datasets run in analysis
        listDatasets = ''
        i = 1
        for each in ds:
            listDatasets = listDatasets+('D'+str(i)+' = '+str(each)+'\n')
            i += 1
        #Report datasets
        analy_report.set_font('Times', 'B', 10)
        analy_report.multi_cell(w = 180, h = 4, txt='Datasets:', border=1, align='L')
        analy_report.y += 1 #Space below section header
        analy_report.set_font('Times','', 8)
        analy_report.multi_cell(w = 180, h = 4, txt=listDatasets, border=1, align='L')
    else:
        analy_report.cell(w = 180, h = 4, txt='Target Training Dataset: '+train_name, border=1, align='L')
        analy_report.y +=5
        analy_report.x = 10

        listDatasets = ''
        i = 1
        for each in ds:
            listDatasets = listDatasets+('D'+str(i)+' = '+str(each)+'\n')
            i += 1
        analy_report.multi_cell(w = 180, h = 4, txt='Applied Datasets: '+'\n'+listDatasets, border=1, align='L')
    footer(analy_report)

    #NEXT PAGE(S) - Exploratory Univariate Analysis for each Dataset------------------------------------------------------------------
    if eval(training):
        print("Publishing Univariate Analysis")
        resultLimit = 5 #Limits to this many dataset results per page
        datasetCount = len(ds)
        #Determine number of pages needed for univariate results
        pageCount = datasetCount / float (resultLimit)
        pageCount = math.ceil(pageCount) #rounds up to next full integer
        for page in range(0, pageCount): #generate each page
            pubUnivariate(analy_report,experiment_path,ds,page,resultLimit,pageCount)

    #NEXT PAGE(S) Data and Model Prediction Summary--------------------------------------------------------------------------------------
    print("Publishing Model Prediction Summary")
    for n in range(len(ds)):
        #Create PDF and Set Options
        analy_report.set_margins(left=1, top=1, right=1, )
        analy_report.add_page()
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt="Dataset and Model Prediction Summary:  D"+str(n+1)+" = "+ds[n], border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=8)

        #Exploratory Analysis ----------------------------
        #Image placement notes:
        #upper left hand coordinates (x,y), then image width then height (image fit to space)
        #upper left hand coordinates (x,y), then image width with hight based on image dimensions (retain original image ratio)
        if eval(training):
            analy_report.image(experiment_path+'/'+ds[n]+'/exploratory/ClassCountsBarPlot.png', 1, 10, 60, 40) #upper left hand coordinates (x,y), then image width then height (image fit to space)
        else:
            analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/exploratory/ClassCountsBarPlot.png', 1, 10, 60, 40) #upper left hand coordinates (x,y), then image width then height (image fit to space)
        try:
            if eval(training):
                analy_report.image(experiment_path+'/'+ds[n]+'/exploratory/FeatureCorrelations.png', 85, 10, 125, 105) #upper left hand coordinates (x,y), then image width with hight based on image dimensions (retain original image ratio)
            else:
                analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/exploratory/FeatureCorrelations.png', 85, 10, 125, 105) #upper left hand coordinates (x,y), then image width with hight based on image dimensions (retain original image ratio)
        except:
            analy_report.x = 125
            analy_report.y = 55
            analy_report.cell(35, 4, 'No Feature Correlation Plot', 1, align="L")
            pass
        if eval(training):
            data_summary = pd.read_csv(experiment_path+'/'+ds[n]+"/exploratory/DataCounts.csv")
        else:
            data_summary = pd.read_csv(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+"/exploratory/DataCounts.csv")
        info_ls = []
        for i in range(len(data_summary)):
            info_ls.append(data_summary.iloc[i,0]+': ')
            info_ls.append(str(data_summary.iloc[i,1]))
            info_ls.append('\n')
        analy_report.x = 1
        analy_report.y = 52
        analy_report.set_font('Times', 'B', 8)
        analy_report.multi_cell(w=60, h=4, txt='Dataset Counts Summary:', border=1, align='L')
        analy_report.set_font('Times','', 8)
        analy_report.multi_cell(w=60, h=4, txt=' '+listToString(info_ls), border=1, align='L')

        #Report Best Algorithms by metric
        if eval(training):
            summary_performance = pd.read_csv(experiment_path+'/'+ds[n]+"/model_evaluation/Summary_performance_mean.csv")
        else:
            summary_performance = pd.read_csv(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+"/model_evaluation/Summary_performance_mean.csv")
        summary_performance['ROC AUC'] = summary_performance['ROC AUC'].astype(float)
        highest_ROC = summary_performance['ROC AUC'].max()
        algorithm = summary_performance[summary_performance['ROC AUC'] == highest_ROC].index.values
        best_alg_ROC =  summary_performance.iloc[algorithm, 0]

        summary_performance['Balanced Accuracy'] = summary_performance['Balanced Accuracy'].astype(float)
        highest_BA = summary_performance['Balanced Accuracy'].max()
        algorithm = summary_performance[summary_performance['Balanced Accuracy'] == highest_BA].index.values
        best_alg_BA =  summary_performance.iloc[algorithm, 0]

        summary_performance['F1 Score'] = summary_performance['F1 Score'].astype(float)
        highest_F1 = summary_performance['F1 Score'].max()
        algorithm = summary_performance[summary_performance['F1 Score'] == highest_F1].index.values
        best_alg_F1 =  summary_performance.iloc[algorithm, 0]

        summary_performance['PRC AUC'] = summary_performance['PRC AUC'].astype(float)
        highest_PRC = summary_performance['PRC AUC'].max()
        algorithm = summary_performance[summary_performance['PRC AUC'] == highest_PRC].index.values
        best_alg_PRC = summary_performance.iloc[algorithm, 0]

        summary_performance['PRC APS'] = summary_performance['PRC APS'].astype(float)
        highest_APS = summary_performance['PRC APS'].max()
        algorithm = summary_performance[summary_performance['PRC APS'] == highest_APS].index.values
        best_alg_APS = summary_performance.iloc[algorithm, 0]

        analy_report.x = 1
        analy_report.y = 85
        analy_report.set_font('Times', 'B', 8)
        analy_report.multi_cell(w=80, h=4, txt='Top ML Algorithm Results (Averaged Over CV Runs):', border=1, align='L')
        analy_report.set_font('Times','', 8)
        analy_report.multi_cell(w=80, h=4, txt="Best (ROC_AUC): "+ str(best_alg_ROC.values)+' = '+ str("{:.3f}".format(highest_ROC))+
                    '\n'+"Best (Balanced Acc.): "+ str(best_alg_BA.values)+' = '+ str("{:.3f}".format(highest_BA))+
                    '\n'+"Best (F1 Score): "+ str(best_alg_F1.values)+' = '+ str("{:.3f}".format(highest_F1))+
                    '\n'+"Best (PRC AUC): "+ str(best_alg_PRC.values)+' = '+ str("{:.3f}".format(highest_PRC))+
                    '\n'+"Best (PRC APS): "+ str(best_alg_APS.values)+' = '+ str("{:.3f}".format(highest_APS)), border=1, align='L')

        analy_report.set_font('Times', 'B', 10)
        #ROC-------------------------------
        analy_report.x = 1
        analy_report.y = 112
        analy_report.cell(10, 4, 'ROC', 1, align="L")
        if eval(training):
            analy_report.image(experiment_path+'/'+ds[n]+'/model_evaluation/Summary_ROC.png', 4, 118, 120)
            analy_report.image(experiment_path+'/'+ds[n]+'/model_evaluation/metricBoxplots/Compare_ROC AUC.png', 124, 118, 82,85)
        else:
            analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/model_evaluation/Summary_ROC.png', 4, 118, 120)
            analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/model_evaluation/metricBoxplots/Compare_ROC AUC.png', 124, 118, 82,85)

        #PRC-------------------------------
        analy_report.x = 1
        analy_report.y = 200
        analy_report.cell(10, 4, 'PRC', 1, align="L")
        if eval(training):
            analy_report.image(experiment_path+'/'+ds[n]+'/model_evaluation/Summary_PRC.png', 4, 206, 133) #wider to account for more text
            analy_report.image(experiment_path+'/'+ds[n]+'/model_evaluation/metricBoxplots/Compare_PRC AUC.png', 138, 205, 68,80)
        else:
            analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/model_evaluation/Summary_PRC.png', 4, 206, 133) #wider to account for more text
            analy_report.image(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/model_evaluation/metricBoxplots/Compare_PRC AUC.png', 138, 205, 68,80)
        footer(analy_report)

    # NEXT PAGE(S) - Average Model Prediction Statistics--------------------------------------------------------------------------------------
    print("Publishing Average Model Prediction Statistics")
    resultLimit = 5 #Limits to this many dataset results per page
    datasetCount = len(ds)
    #Determine number of pages needed for univariate results
    pageCount = datasetCount / float (resultLimit)
    pageCount = math.ceil(pageCount) #rounds up to next full integer
    analy_report.set_fill_color(200)
    for page in range(0, pageCount): #generate each page
        pubModelStats(analy_report,experiment_path,ds,page,resultLimit,pageCount,training,train_name)

    #NEXT PAGE(S) - ML Dataset Feature Importance Summary----------------------------------------------------------------
    if eval(training):
        print("Publishing Feature Importance Summaries")
        for k in range(len(ds)):
            analy_report.add_page()
            analy_report.set_font('Times', 'B', 12)
            analy_report.cell(w=0, h = 8, txt="Feature Importance Summary:  D"+str(k+1) +' = '+ ds[k], border=1, align="L", ln=2)
            analy_report.set_font(family='times', size=9)
            analy_report.image(experiment_path+'/'+ds[k]+'/feature_selection/mutualinformation/TopAverageScores.png', 5, 12, 100,135) #Images adjusted to fit a width of 100 and length of 135
            analy_report.image(experiment_path+'/'+ds[k]+'/feature_selection/multisurf/TopAverageScores.png', 105, 12, 100,135)
            analy_report.x = 0
            analy_report.y = 150
            analy_report.cell(0, 8, "Composite Feature Importance Plot (Normalized and Performance Weighted)", 1, align="L")
            analy_report.image(experiment_path+'/'+ds[k]+'/model_evaluation/feature_importance/Compare_FI_Norm_Weight.png', 1, 159, 208, 125) #130 added
            footer(analy_report)

    #NEXT PAGE - Create Dataset Boxplot Comparison Page---------------------------------------
    if eval(training):
        print("Publishing Dataset Comparison Boxplots")
        analy_report.add_page()
        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt="Compare ML Performance Across Datasets", border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=9)
        if len(ds) > 1:
            analy_report.image(experiment_path+'/DatasetComparisons/dataCompBoxplots/'+'DataCompareAllModels_ROC AUC.png',  1, 12, 208, 130) #Images adjusted to fit a width of 100 and length of 135
            analy_report.image(experiment_path+'/DatasetComparisons/dataCompBoxplots/'+'DataCompareAllModels_PRC AUC.png',  1, 150, 208, 130) #Images adjusted to fit a width of 100 and length of 135
        footer(analy_report)

    #NEXT PAGE(S) -Create Best Kruskall Wallis Dataset Comparison Page---------------------------------------
    if eval(training):
        print("Publishing Statistical Analysis")
        analy_report.add_page(orientation='P')
        analy_report.set_margins(left=1, top=10, right=1, )

        d = []
        for i in range(len(ds)):
            d.append('Data '+str(i+1)+'= '+ ds[i])
            d.append('\n')

        analy_report.set_font('Times', 'B', 12)
        analy_report.cell(w=0, h = 8, txt='Using Best Performing Algorithms (Kruskall Wallis Compare Datasets)', border=1, align="L", ln=2)
        analy_report.set_font(family='times', size=7)

        #Dataset list Key
        listDatasets = ''
        i = 1
        for each in ds:
            listDatasets = listDatasets+('D'+str(i)+' = '+str(each)+'\n')
            i += 1
        analy_report.x = 5
        analy_report.y = 14
        analy_report.multi_cell(w = 0, h = 4, txt='Datasets: '+'\n'+listDatasets, border=1, align='L')
        analy_report.y += 5

        success = False
        try:
            #Kruskall Wallis Table
            #A table can take at most 4 datasets to fit comfortably with these settings
            kw_ds = pd.read_csv(experiment_path+'/DatasetComparisons/'+'BestCompare_KruskalWallis.csv',sep=',',index_col=0)
            kw_ds = kw_ds.round(4)
            success = True
        except:
            pass

        if success:
            #Process
            for i in range(len(ds)):
                kw_ds = kw_ds.drop('Std_D'+str(i+1),1)
            kw_ds = kw_ds.drop('Statistic',1)
            kw_ds = kw_ds.drop('Sig(*)',1)

            #Format
            kw_ds.reset_index(inplace=True)
            kw_ds = kw_ds.columns.to_frame().T.append(kw_ds, ignore_index=True)
            kw_ds.columns = range(len(kw_ds.columns))
            epw = 208 #Amount of Space (width) Avaliable
            th = analy_report.font_size
            #col_width = epw/float(10) #maximum column width
            col_width_list = [23,14,30,14,30,14,30,14]

            dfLength = len(ds)
            if len(ds) <= 3: #4
                col_count = 0
                kw_ds = kw_ds.to_numpy()
                for row in kw_ds:
                    for datum in row:
                        analy_report.cell(col_width_list[col_count], th, str(datum), border=1)
                        col_count +=1
                    col_count = 0
                    analy_report.ln(th) #critical
            else:
                #Print next 3 datasets
                col_count = 0
                table1 = kw_ds.iloc[: , :8] #10
                table1 = table1.to_numpy()
                for row in table1:
                    for datum in row:
                        analy_report.cell(col_width_list[col_count], th, str(datum), border=1)
                        col_count +=1
                    col_count = 0
                    analy_report.ln(th) #critical
                analy_report.y += 5

                col_count = 0
                table1 = kw_ds.iloc[: , 8:14] #10:18
                met = kw_ds.iloc[:,0]
                met2 = kw_ds.iloc[:,1]
                table1 = pd.concat([met,met2, table1], axis=1)
                table1 = table1.to_numpy()
                for row in table1:
                    for datum in row:
                        analy_report.cell(col_width_list[col_count], th, str(datum), border=1)
                        col_count +=1
                    col_count = 0
                    analy_report.ln(th) #critical
                analy_report.y += 5

                if len(ds) > 6: #8
                    col_count = 0
                    table1 = kw_ds.iloc[: , 14:20] #18:26
                    met = kw_ds.iloc[:,0]
                    met2 = kw_ds.iloc[:,1]
                    table1 = pd.concat([met,met2,table1], axis=1)
                    table1 = table1.to_numpy()
                    for row in table1:
                        for datum in row:
                            analy_report.cell(col_width_list[col_count], th, str(datum), border=1)
                            col_count +=1
                        col_count = 0
                        analy_report.ln(th) #critical
                    analy_report.y += 5

                if len(ds) > 9:
                    table1 = kw_ds.iloc[: , 20:26]
                    met = kw_ds.iloc[:,0]
                    met2 = kw_ds.iloc[:,1]
                    table1 = pd.concat([met,met2,table1], axis=1)
                    table1 = table1.to_numpy()
                    for row in table1:
                        for datum in row:
                            analy_report.cell(col_width_list[col_count], th, str(datum), border=1)
                            col_count +=1
                        col_count = 0
                        analy_report.ln(th) #critical
                    analy_report.y += 5

                if len(ds) > 12:
                    analy_report.x = 0
                    analy_report.y = 260
                    analy_report.cell(0, 4, 'Warning: Additional dataset results could not be displayed', 1, align="C")

        footer(analy_report)

    #LAST PAGE - Create Runtime Summary Page---------------------------------------
    if eval(training):
        print("Publishing Runtime Summary")
        resultLimit = 8 #Limits to this many dataset results per page
        datasetCount = len(ds)
        #Determine number of pages needed for univariate results
        pageCount = datasetCount / float (resultLimit)
        pageCount = math.ceil(pageCount) #rounds up to next full integer
        for page in range(0, pageCount): #generate each page
            pubRuntime(analy_report,experiment_path,ds,page,resultLimit,pageCount)

    #Output The PDF Object
    try:
        if eval(training):
            fileName = str(experiment_name)+'_ML_Pipeline_Report.pdf'
            analy_report.output(experiment_path+'/'+fileName)
            # Print phase completion
            print("Phase 8 complete")
            try:
                job_file = open(experiment_path + '/jobsCompleted/job_data_pdf_training.txt', 'w')
                job_file.write('complete')
                job_file.close()
            except:
                pass
        else:
            fileName = str(experiment_name)+'_ML_Pipeline_Apply_Report.pdf'
            analy_report.output(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/'+fileName)
            # Print phase completion
            print("Phase 10 complete")
            try:
                job_file = open(experiment_path + '/jobsCompleted/job_data_pdf_apply_'+str(train_name) +'.txt', 'w')
                job_file.write('complete')
                job_file.close()
            except:
                pass
    except:
        print('Pdf Output Failed')

def pubUnivariate(analy_report,experiment_path,ds,page,resultLimit,pageCount):
    """ Generates single page of univariate analysis results. Automatically moves to another page when runs out of space. Maximum of 4 dataset results to a page. """
    datasetCount = len(ds)
    dataStart = page*resultLimit
    countLimit = (page*resultLimit)+resultLimit
    analy_report.add_page(orientation='P')
    analy_report.set_font('Times', 'B', 12)
    if pageCount > 1:
        analy_report.cell(w=180, h = 8, txt='Univariate Analysis of Each Dataset (Top 10 Features for Each): Page '+str(page+1), border=1, align='L', ln=2)
    else:
        analy_report.cell(w=180, h = 8, txt='Univariate Analysis of Each Dataset (Top 10 Features for Each)', border=1, align='L', ln=2)
    for n in range(dataStart,datasetCount):
        if n >= countLimit: #Stops generating page when dataset count limit reached
            break
        analy_report.y += 2
        sig_df = pd.read_csv(experiment_path+'/'+ds[n]+'/exploratory/univariate_analyses/Univariate_Significance.csv')
        sig_ls = []
        sig_df = sig_df.nsmallest(10, ['p-value'])
        for i in range(len(sig_df)):
            sig_ls.append(sig_df.iloc[i,0]+': ')
            sig_ls.append(str(sig_df.iloc[i,1]))
            sig_ls.append('\n')
        analy_report.set_font('Times', 'B', 10)
        analy_report.multi_cell(w=180, h=4, txt='D'+str(n+1)+' = '+ds[n], border=1, align='L')
        analy_report.y += 1 #Space below section header
        analy_report.set_font('Times','B', 8)
        analy_report.multi_cell(w=180, h=4, txt='Feature:  P-Value', border=1, align='L')
        analy_report.set_font('Times','', 8)
        analy_report.multi_cell(w=180, h=4, txt=' '+listToString(sig_ls), border=1, align='L')
    footer(analy_report)


def pubModelStats(analy_report,experiment_path,ds,page,resultLimit,pageCount,training,train_name):
    datasetCount = len(ds)
    dataStart = page*resultLimit
    countLimit = (page*resultLimit)+resultLimit
    #Create PDF and Set Options
    analy_report.set_margins(left=1, top=1, right=1, )
    analy_report.add_page()
    analy_report.set_font('Times', 'B', 12)
    if pageCount > 1:
        analy_report.cell(w=0, h = 8, txt='Average Model Prediction Statistics (Rounded to 3 Decimal Points): Page '+str(page+1), border=1, align='L', ln=2)
    else:
        analy_report.cell(w=0, h = 8, txt='Average Model Prediction Statistics (Rounded to 3 Decimal Points)', border=1, align='L', ln=2)
    for n in range(dataStart,datasetCount):
        if n >= countLimit: #Stops generating page when dataset count limit reached
            break
        analy_report.y += 4
        analy_report.set_font('Times', 'B', 10)
        analy_report.multi_cell(w=0, h=4, txt='D'+str(n+1)+' = '+ds[n], border=1, align='L')
        analy_report.y += 1 #Space below section header
        analy_report.set_font('Times','', 7)
        if eval(training):
            stats_ds = pd.read_csv(experiment_path+'/'+str(ds[n])+'/model_evaluation/Summary_performance_mean.csv',sep=',',index_col=0)
        else:
            stats_ds = pd.read_csv(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/model_evaluation/Summary_performance_mean.csv',sep=',',index_col=0)
        #Make list of top values for each metric
        metricNameList = ['Balanced Accuracy','Accuracy','F1 Score','Sensitivity (Recall)','Specificity','Precision (PPV)','TP','TN','FP','FN','NPV','LR+','LR-','ROC AUC','PRC AUC','PRC APS']
        bestMetricList = []
        if eval(training):
            ds2 = pd.read_csv(experiment_path+'/'+ds[n]+"/model_evaluation/Summary_performance_mean.csv")
        else:
            ds2 = pd.read_csv(experiment_path+'/'+train_name+'/applymodel/'+ds[n]+'/model_evaluation/Summary_performance_mean.csv')
        for metric in metricNameList:
            ds2[metric] = ds2[metric].astype(float).round(3)
            metricMax = ds2[metric].max()
            bestMetricList.append(metricMax)

        stats_ds = stats_ds.round(3)

        #Format
        stats_ds.reset_index(inplace=True)
        stats_ds = stats_ds.columns.to_frame().T.append(stats_ds, ignore_index=True)
        stats_ds.columns = range(len(stats_ds.columns))
        epw = 208 #Amount of Space (width) Avaliable
        th = analy_report.font_size
        col_width_list = [32,11,11,8,12,12,10,15,15,15,15,8,9,9,8,8,8]
        table1 = stats_ds.iloc[: , :18]
        table1 = table1.to_numpy()

        #Print table header first
        row_count = 0
        col_count = 0
        for row in table1: # each row
            if row_count == 0:
                #Print first row
                for datum in row:
                    if col_count == 0:
                        analy_report.cell(col_width_list[col_count], th, 'ML Algorithm', border=0, align="C")
                    else:
                        entryList = str(datum).split(' ')
                        analy_report.cell(col_width_list[col_count], th, entryList[0], border=0, align="C")
                    col_count +=1
                analy_report.ln(th) #critical
                col_count = 0
                #Print second row
                for datum in row:
                    entryList = str(datum).split(' ')
                    try:
                        analy_report.cell(col_width_list[col_count], th, entryList[1], border=0, align="C")
                    except:
                        analy_report.cell(col_width_list[col_count], th, ' ', border=0, align="C")
                    col_count +=1
                analy_report.ln(th) #critical
                col_count = 0
            else: #Print table contents
                for datum in row: #each column
                    if col_count > 0 and float(datum) == float(bestMetricList[col_count-1]):
                        analy_report.cell(col_width_list[col_count], th, str(datum), border=1, align="L", fill=True)
                    else:
                        analy_report.cell(col_width_list[col_count], th, str(datum), border=1, align="L")
                    col_count +=1
                analy_report.ln(th) #critical
                col_count = 0
            row_count += 1
    footer(analy_report)


def pubRuntime(analy_report,experiment_path,ds,page,resultLimit,pageCount):
    """ Generates single page of univariate analysis results. Automatically moves to another page when runs out of space. Maximum of 4 dataset results to a page. """
    col_width = 40 #maximum column width
    datasetCount = len(ds)
    dataStart = page*resultLimit
    countLimit = (page*resultLimit)+resultLimit
    analy_report.add_page(orientation='P')
    analy_report.set_font('Times', 'B', 12)
    if pageCount > 1:
        analy_report.cell(w=0, h = 8, txt='Pipeline Runtime Summary: Page '+str(page+1), border=1, align='L', ln=2)
    else:
        analy_report.cell(w=0, h = 8, txt='Pipeline Runtime Summary', border=1, align='L', ln=2)
    analy_report.set_font('Times','', 8)
    th = analy_report.font_size
    analy_report.y += 2
    left = True
    for n in range(dataStart,datasetCount):
        if n >= countLimit: #Stops generating page when dataset count limit reached
            break
        lastY = analy_report.y
        lastX = analy_report.x
        if left:
            analy_report.x = 1
        time_df = pd.read_csv(experiment_path+'/'+ds[n]+'/runtimes.csv')
        time_df.iloc[:, 1] = time_df.iloc[:, 1].round(2)
        time_df = time_df.columns.to_frame().T.append(time_df, ignore_index=True)
        time_df = time_df.to_numpy()
        analy_report.set_font('Times', 'B', 10)
        analy_report.cell(col_width*2, 4, str(ds[n]), 1, align="L")
        analy_report.y += 5
        analy_report.x = lastX
        analy_report.set_font('Times','', 7)
        for row in time_df:
            for datum in row:
                analy_report.cell(col_width, th, str(datum), border=1)
            analy_report.ln(th) #critical
            analy_report.x = lastX

        if left:
            analy_report.x = (col_width*2)+2
            analy_report.y = lastY
            left = False
        else:
            analy_report.x = 1
            analy_report.y = lastY+63
            left = True
    footer(analy_report)


def listToString(s):
    str1 = " "
    return (str1.join(s))


#Create Footer
def footer(self):
    self.set_auto_page_break(auto=False, margin=3)
    self.set_y(285)
    self.set_font('Times', 'I', 7)
    self.cell(0, 7,'Generated with the URBS-Lab STREAMLINE: (https://github.com/UrbsLab/STREAMLINE)', 0, 0, 'C')
    self.set_font(family='times', size=9)

#Find N greatest ingegers within a list
def ngi(list1, N):
    final_list = []
    for i in range(0, N):
        max1 = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j];
        list1.remove(max1);
        final_list.append(max1)

if __name__ == '__main__':
    job(sys.argv[1],sys.argv[2],sys.argv[3],sys.argv[4])
