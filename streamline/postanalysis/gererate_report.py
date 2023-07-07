import glob
import logging
import math
import os
import pickle
import csv
from datetime import datetime
from pathlib import Path

from streamline import __version__ as version
import pandas as pd
from fpdf import FPDF

from streamline.modeling.utils import is_supported_model
from streamline.modeling.utils import REGRESSION_ABBREVIATION, REGRESSION_COLORS, SUPPORTED_REGRESSION_MODELS
from streamline.utils.job import Job


class ReportJob(Job):
    """
    This 'Job' script is called by PDF_ReportMain.py which generates a formatted PDF summary report of key
    pipeline results It is run once for the whole pipeline analysis.
    """

    def __init__(self, output_path=None, experiment_name=None, experiment_path=None, algorithms=None,
                 exclude=("XCS", "eLCS"),
                 training=True, data_path=None, rep_data_path=None, load_algo=True):
        super().__init__()
        self.time = None
        assert (output_path is not None and experiment_name is not None) or (experiment_path is not None)
        if output_path is not None and experiment_name is not None:
            self.output_path = output_path
            self.experiment_name = experiment_name
            self.experiment_path = self.output_path + '/' + self.experiment_name
        else:
            self.experiment_path = experiment_path
            self.experiment_name = self.experiment_path.split('/')[-1]
            self.output_path = self.experiment_path.split('/')[-2]

        self.training = training

        self.train_name = None
        # Find folders inside directory
        if self.training:
            self.datasets = os.listdir(self.experiment_path)
            remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle',
                           'DatasetComparisons', 'jobs', 'jobsCompleted', 'logs',
                           'KeyFileCopy', 'dask_logs',
                           experiment_name + '_ML_Pipeline_Report.pdf']
            for item in remove_list:
                if item in self.datasets:
                    self.datasets.remove(item)
            if '.idea' in self.datasets:
                self.datasets.remove('.idea')
            self.datasets = sorted(self.datasets)
        else:
            self.train_name = data_path.split('/')[-1].split('.')[0]
            self.datasets = []
            for dataset_filename in glob.glob(rep_data_path + '/*'):
                dataset_filename = str(Path(dataset_filename).as_posix())
                # dataset_filename = str(dataset_filename).replace('\\', '/')
                # Save unique dataset names so that analysis is run only once if there is both a
                # .txt and .csv version of dataset with same name.
                apply_name = dataset_filename.split('/')[-1].split('.')[0]
                self.datasets.append(apply_name)
            self.datasets = sorted(self.datasets)

        dataset_directory_paths = []
        for dataset in self.datasets:
            full_path = self.experiment_path + "/" + dataset
            dataset_directory_paths.append(full_path)

        self.dataset_directory_paths = dataset_directory_paths

        if algorithms is None:
            self.algorithms = SUPPORTED_REGRESSION_MODELS
            if exclude is not None:
                for algorithm in exclude:
                    try:
                        self.algorithms.remove(algorithm)
                    except Exception:
                        Exception("Unknown algorithm in exclude: " + str(algorithm))
        else:
            self.algorithms = list()
            for algorithm in algorithms:
                self.algorithms.append(is_supported_model(algorithm))

        # Unpickle metadata from previous phase
        file = open(self.experiment_path + '/' + "metadata.pickle", 'rb')
        self.metadata = pickle.load(file)
        file.close()

        file = open(self.experiment_path + '/' + "algInfo.pickle", 'rb')
        self.alg_info = pickle.load(file)
        file.close()
        # self.metadata = {}

        if load_algo:
            temp_algo = []
            for key in self.alg_info:
                if self.alg_info[key][0]:
                    temp_algo.append(key)
            self.algorithms = temp_algo

        self.abbrev = dict((k, REGRESSION_ABBREVIATION[k]) for k in self.algorithms if k in REGRESSION_ABBREVIATION)
        self.colors = dict((k, REGRESSION_COLORS[k]) for k in self.algorithms if k in REGRESSION_COLORS)
        self.metrics = None

        self.analysis_report = FPDF('P', 'mm', 'A4')

    def run(self):
        self.job()

    def job(self):

        self.job_start_time = datetime.now()
        self.time = datetime.now()

        # Turn metadata dictionary into text list
        ars_dic = []
        for key in self.metadata:
            ars_dic.append(str(key) + ':')
            ars_dic.append(str(self.metadata[key]))
            ars_dic.append('\n')

        # Turn alg_info dictionary into text list
        ars_dic_2 = []
        for key in sorted(self.alg_info.keys()):
            ars_dic_2.append(str(key) + ':')
            ars_dic_2.append(str(self.alg_info[key][0]))
            ars_dic_2.append('\n')

        # Analysis Settings, Global Analysis Settings, ML Modeling Algorithms
        self.analysis_report.set_margins(left=10, top=5, right=10, )
        self.analysis_report.add_page(orientation='P')

        # PDF page dimension reference
        # page width = 210 and page height down to start of footer = 285 (these are estimates)
        # FRONT PAGE - Summary of Pipeline settings
        # -------------------------------------------------------------------------------------------------------
        logging.info("Starting Report")
        inc = 1
        targetdata = ars_dic[0:21+inc]  # Data-path to  instance label
        cv = ars_dic[21+inc:27+inc]  # cv partitions to partition Method
        match = ars_dic[27+inc:30+inc]  # match label
        cat_cut = ars_dic[30+inc:33+inc]  # categorical cutoff
        stat_cut = ars_dic[33+inc:36+inc]  # statistical significance cutoff
        process = ars_dic[36+inc:51+inc]  # feature missingness cutoff to list of exploratory plots saved
        general = ars_dic[51+inc:57+inc]  # random seed to run from notebooks
        process2 = ars_dic[57+inc:66+inc]  # use data scaling to use multivariate imputation
        featsel = ars_dic[66+inc:93+inc]  # use mutual info to export feature importance plots
        overwrite = ars_dic[93+inc:96+inc]  # overwrite cv
        modeling = ars_dic[96+inc:114+inc]  # primary metric to export hyperparameter sweep plots
        lcs = ars_dic[114+inc:129+inc]
        stats = ars_dic[129+inc:150+inc]

        ls2 = ars_dic_2

        self.analysis_report.set_font('Times', 'B', 12)
        if self.training:
            self.analysis_report.cell(w=180, h=8, txt='STREAMLINE Testing Data Evaluation Report: ' + str(self.time), ln=2,
                                      border=1, align='L')
        else:
            self.analysis_report.cell(w=180, h=8, txt='STREAMLINE Replication Data Evaluation Report: ' + str(self.time),
                                      ln=2, border=1, align='L')

        self.analysis_report.y += 2  # Margin below page header

        if self.training:
            # Get names of self.datasets run in analysis
            list_datasets = ''
            i = 1
            for each in self.datasets:
                list_datasets = list_datasets + ('D' + str(i) + ' = ' + str(each) + '\n')
                i += 1
            # Report self.datasets
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.multi_cell(w=180, h=4, txt='Target Dataset(s)', border=1, align='L')
            self.analysis_report.y += 1  # Space below section header
            self.analysis_report.set_font('Times', '', 8)
            self.analysis_report.multi_cell(w=180, h=4, txt=list_datasets, border=1, align='L')
        else:
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.cell(w=180, h=4, txt='Target Training Dataset: ' + self.train_name, border=1,
                                      align='L')
            self.analysis_report.y += 5
            self.analysis_report.x = 10

            list_datasets = ''
            i = 1
            for each in self.datasets:
                list_datasets = list_datasets + ('D' + str(i) + ' = ' + str(each) + '\n')
                i += 1
            self.analysis_report.multi_cell(w=180, h=4, txt='Applied to Following Replication Dataset(s): ', border=1, align='L')
            self.analysis_report.y += 1  # Space below section header

            self.analysis_report.set_font('Times', '', 8)
            self.analysis_report.multi_cell(w=180, h=4, txt= list_datasets, border=1, align='L')
            #self.analysis_report.multi_cell(w=180, h=4, txt='Applied to Following Replication Dataset(s): ' + '\n' + list_datasets, border=1, align='L')

        self.analysis_report.y += 2  # Margin below Datasets

        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.cell(w=180, h=4, txt='STREAMLINE Run Settings', ln=2, border=1, align='L')

        self.analysis_report.y += 1  # Margin below page header
        top_of_list = self.analysis_report.y  # Page height for start of algorithm settings
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='General Pipeline Settings:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.multi_cell(w=90, h=4,
                                        txt=' ' + list_to_string(cv) + ' ' + list_to_string(
                                            cat_cut) + ' ' + list_to_string(stat_cut) + ' ' + list_to_string(
                                            general),
                                        border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='EDA and Processing Settings:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.multi_cell(w=90, h=4,
                                        txt=' ' + list_to_string(process) + ' ' + list_to_string(
                                            process2) + ' ' + list_to_string(overwrite),
                                        border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='Feature Importance/Selection Settings:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.multi_cell(w=90, h=4,
                                        txt=' ' + list_to_string(featsel),
                                        border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='Target Data Settings:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.multi_cell(w=90, h=4,
                                        txt=' ' + list_to_string(targetdata),
                                        border=1, align='L')

        #bottom_of_list = self.analysis_report.y
        #self.analysis_report.y = bottom_of_list + 2

        self.analysis_report.x += 90
        self.analysis_report.y = top_of_list  # 96
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='ML Modeling Algorithms:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.x += 90
        self.analysis_report.multi_cell(w=90, h=4, txt=' ' + list_to_string(ls2), border=1, align='L')
        self.analysis_report.y += 1

        self.analysis_report.x += 90
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='Modeling Settings:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.x += 90
        self.analysis_report.multi_cell(w=90, h=4, txt=' ' + list_to_string(modeling), border=1, align='L')
        self.analysis_report.y += 1

        self.analysis_report.x += 90
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='LCS Settings (eLCS,XCS,ExSTraCS):', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.x += 90
        self.analysis_report.multi_cell(w=90, h=4, txt=' ' + list_to_string(lcs), border=1, align='L')
        self.analysis_report.y += 1

        self.analysis_report.x += 90
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='Stats and Figure Settings:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.x += 90
        self.analysis_report.multi_cell(w=90, h=4, txt=' ' + list_to_string(stats), border=1, align='L')



        """
        try_again = True
        try:
            self.analysis_report.image('info/Pictures/STREAMLINE_LOGO.png', 102, 150, 90)
            try_again = False
        except Exception:
            pass
        if try_again:
            try:  # Running on Google Colab
                self.analysis_report.image('/content/drive/MyDrive/STREAMLINE/info/Pictures/STREAMLINE_LOGO.png', 102, 150,
                                           90)
            except Exception:
                pass
        """



        """
        ls1 = ars_dic[0:87]  # DataPath to OverwriteCVDatasets - filter poor [0:87]
        # ls2 = ars_dic[87:132]  # ML modeling algorithms (NaiveB - ExSTraCS) [87:132]
        ls2 = ars_dic_2
        ls3 = ars_dic[87:105]  # primary metric - Export Hyperparameter SweepPLot  [132:150]
        ls4 = ars_dic[105:129]  # DoLCS Hyperparameter Sweep LCS hyper-sweep timeout) [150:165]
        ls5 = ars_dic[129:147]  # ExportROCPlot to Top Model Features to Display [165:180]

        self.analysis_report.set_font('Times', 'B', 12)
        if self.training:
            self.analysis_report.cell(w=180, h=8, txt='STREAMLINE Testing Evaluation Report: ' + str(self.time), ln=2,
                                      border=1, align='L')
        else:
            self.analysis_report.cell(w=180, h=8, txt='STREAMLINE Replication Evaluation Report: ' + str(self.time),
                                      ln=2, border=1, align='L')
        self.analysis_report.y += 2  # Margin below page header
        top_of_list = self.analysis_report.y  # Page height for start of algorithm settings
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='General Pipeline Settings:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.multi_cell(w=90, h=4,
                                        txt=' ' + list_to_string(ls1) + ' ' + list_to_string(
                                            ls3) + ' ' + list_to_string(
                                            ls5),
                                        border=1, align='L')
        bottom_of_list = self.analysis_report.y
        self.analysis_report.x += 90
        self.analysis_report.y = top_of_list  # 96
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='ML Modeling Algorithms:', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.x += 90
        self.analysis_report.multi_cell(w=90, h=4, txt=' ' + list_to_string(ls2), border=1, align='L')
        self.analysis_report.x += 90
        self.analysis_report.y += 2
        self.analysis_report.set_font('Times', 'B', 10)
        self.analysis_report.multi_cell(w=90, h=4, txt='LCS Settings (eLCS,XCS,ExSTraCS):', border=1, align='L')
        self.analysis_report.y += 1  # Space below section header
        self.analysis_report.set_font('Times', '', 8)
        self.analysis_report.x += 90
        self.analysis_report.multi_cell(w=90, h=4, txt=' ' + list_to_string(ls4), border=1, align='L')
        self.analysis_report.y = bottom_of_list + 2

        try_again = True
        try:
            self.analysis_report.image('info/Pictures/STREAMLINE_LOGO.png', 102, 150, 90)
            try_again = False
        except Exception:
            pass
        if try_again:
            try:  # Running on Google Colab
                self.analysis_report.image('/content/drive/MyDrive/STREAMLINE/info/Pictures/STREAMLINE_LOGO.png', 102, 150,
                                           90)
            except Exception:
                pass

        if self.training:
            # Get names of self.datasets run in analysis
            list_datasets = ''
            i = 1
            for each in self.datasets:
                list_datasets = list_datasets + ('D' + str(i) + ' = ' + str(each) + '\n')
                i += 1
            # Report self.datasets
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.multi_cell(w=180, h=4, txt='Datasets', border=1, align='L')
            self.analysis_report.y += 1  # Space below section header
            self.analysis_report.set_font('Times', '', 8)
            self.analysis_report.multi_cell(w=180, h=4, txt=list_datasets, border=1, align='L')
        else:
            self.analysis_report.cell(w=180, h=4, txt='Target Training Dataset: ' + self.train_name, border=1,
                                      align='L')
            self.analysis_report.y += 5
            self.analysis_report.x = 10

            list_datasets = ''
            i = 1
            for each in self.datasets:
                list_datasets = list_datasets + ('D' + str(i) + ' = ' + str(each) + '\n')
                i += 1
            self.analysis_report.multi_cell(w=180, h=4, txt='Applied self.datasets: ' + '\n' + list_datasets, border=1,
                                            align='L')
        """
        self.footer()

        # NEXT PAGE(S) - Exploratory Univariate Analysis for each Dataset
        # ------------------------------------------------------------------
        if self.training:
            logging.info("Publishing Univariate Analysis")
            result_limit = 5  # Limits to this many dataset results per page
            dataset_count = len(self.datasets)
            # Determine number of pages needed for univariate results
            page_count = dataset_count / float(result_limit)
            page_count = math.ceil(page_count)  # rounds up to next full integer
            for page in range(0, page_count):  # generate each page
                self.pub_univariate(page, result_limit, page_count)

        # NEXT PAGE(S) Data and Model Prediction Summary
        # --------------------------------------------------------------------------------------
        M = None
        logging.info("Publishing Model Prediction Summary")
        for m in range(len(self.datasets)):
            M = m
            # Create PDF and Set Options
            self.analysis_report.set_margins(left=1, top=1, right=1, )
            self.analysis_report.add_page()
            self.analysis_report.set_font('Times', 'B', 12)
            self.analysis_report.cell(w=0, h=8,
                                      txt="Dataset and Model Prediction Summary:  D" + str(m + 1) + " = " +
                                          self.datasets[m],
                                      border=1, align="L", ln=2)
            self.analysis_report.set_font(family='times', size=8)

            # Exploratory Analysis ----------------------------
            # Image placement notes:
            # upper left hand coordinates (x,y), then image width then height (image fit to space)
            # upper left hand coordinates (x,y), then image width with height based on image dimensions
            # (retain original image ratio)

            # Insert Data Processing Count Summary
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.x = 1
            self.analysis_report.y = 10
            self.analysis_report.cell(119, 4, 'Data Processing/Counts Summary', 1, align="L")

            self.analysis_report.x = 1
            self.analysis_report.y = 15
            self.analysis_report.set_font('Times', '', 7)
            self.analysis_report.set_fill_color(200)

            if self.training:
                data_process_path = self.experiment_path + '/' + self.datasets[
                    m] + "/exploratory/DataProcessSummary.csv"
            else:
                data_process_path = self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                    m] + "/exploratory/DataProcessSummary.csv"

            table1 = []  # Initialize an empty list to store the data

            with open(data_process_path, "r") as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    table1.append(row)
            # Format
            # data_summary = data_summary.round(3)
            th = self.analysis_report.font_size
            col_width_list = [13, 13, 13, 14, 14, 13, 13, 13, 13]  # 91 x space total

            # Print table header first
            row_count = 0
            col_count = 0
            previous_row = None

            for row in table1:  # each row
                # Make header
                if row_count == 0:
                    for datum in row:  # Print first row
                        entry_list = str(datum).split(' ')
                        self.analysis_report.cell(col_width_list[col_count], th, entry_list[0], border=0, align="C")
                        col_count += 1
                    self.analysis_report.ln(th)  # critical
                    col_count = 0
                    for datum in row:  # Print second row
                        entry_list = str(datum).split(' ')
                        try:
                            self.analysis_report.cell(col_width_list[col_count], th, entry_list[1], border=0, align="C")
                        except Exception:
                            self.analysis_report.cell(col_width_list[col_count], th, ' ', border=0, align="C")
                        col_count += 1
                    self.analysis_report.ln(th)  # critical
                    col_count = 0
                # Fill in data
                elif row_count == 1:
                    previous_row = row
                    for datum in row:
                        if col_count == 0:
                            self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1, align="L",
                                                      fill=True)
                        elif col_count == 6:  # missing percent column
                            self.analysis_report.cell(col_width_list[col_count], th, str(round(float(datum), 4)),
                                                      border=1, align="L", fill=True)
                        else:
                            self.analysis_report.cell(col_width_list[col_count], th, str(int(float(datum))), border=1,
                                                      align="L", fill=True)
                        col_count += 1
                    self.analysis_report.ln(th)  # critical
                    col_count = 0
                else:
                    for datum in row:
                        if col_count == 0:
                            self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1, align="L")
                        elif str(previous_row[col_count]) == str(row[col_count]):  # Value unchanged
                            if col_count == 6:  # missing percent column
                                self.analysis_report.cell(col_width_list[col_count], th, str(round(float(datum), 4)),
                                                          border=1, align="L")
                            else:
                                self.analysis_report.cell(col_width_list[col_count], th, str(int(float(datum))),
                                                          border=1, align="L")
                        else:
                            if col_count == 6:  # missing percent column
                                self.analysis_report.cell(col_width_list[col_count], th, str(round(float(datum), 4)),
                                                          border=1, align="L", fill=True)
                            else:
                                self.analysis_report.cell(col_width_list[col_count], th, str(int(float(datum))),
                                                          border=1, align="L", fill=True)
                        col_count += 1
                    self.analysis_report.ln(th)  # critical
                    col_count = 0
                    previous_row = row
                row_count += 1
            row_count -= 1
            for datum in table1[row_count]:
                if col_count == 0:
                    self.analysis_report.cell(col_width_list[col_count], th, 'Processed', border=1, align="L",
                                              fill=True)
                else:
                    if col_count == 6:  # missing percent column
                        self.analysis_report.cell(col_width_list[col_count], th, str(round(float(datum), 4)), border=1,
                                                  align="L", fill=True)
                    else:
                        self.analysis_report.cell(col_width_list[col_count], th, str(int(float(datum))), border=1,
                                                  align="L", fill=True)
                col_count += 1

            self.analysis_report.set_font('Times', 'B', 8)
            self.analysis_report.x = 1
            self.analysis_report.y = 41
            self.analysis_report.cell(90, 4, 'Cleaning (C) and Engineering (E) Elements', 0, align="L")
            self.analysis_report.set_font('Times', '', 7)
            self.analysis_report.ln(th)  # critical
            self.analysis_report.cell(90, 4, ' * C1 - Remove instances with no outcome and features to ignore', 0, align="L")
            self.analysis_report.ln(th)  # critical
            self.analysis_report.cell(90, 4, ' * E1 - Add missingness features', 0, align="L")
            self.analysis_report.ln(th)  # critical
            self.analysis_report.cell(90, 4, ' * C2 - Remove features with high missingness', 0, align="L")
            self.analysis_report.ln(th)  # critical
            self.analysis_report.cell(90, 4, ' * C3 - Remove instances with high missingness', 0, align="L")
            self.analysis_report.ln(th)  # critical
            self.analysis_report.cell(90, 4, ' * E2 - Add one-hot-encoding of categorical features', 0, align="L")
            self.analysis_report.ln(th)  # critical
            self.analysis_report.cell(90, 4, ' * C4 - Remove highly correlated features', 0, align="L")

            # Insert Class Imbalance barplot
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.x = 70
            self.analysis_report.y = 42
            self.analysis_report.cell(45, 4, 'Class Balance (Processed)', 1, align="L")
            self.analysis_report.set_font('Times', '', 8)
            if self.training:
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[m] + '/exploratory/ClassCountsBarPlot.png', 68, 47, 45,
                    35)
                # upper left hand coordinates (x,y), then image width then height (image fit to space)
            else:
                self.analysis_report.image(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                        m] + '/exploratory/ClassCountsBarPlot.png', 68, 47, 45, 35)
                # upper left hand coordinates (x,y), then image width then height (image fit to space)

            # Insert Feature Correlation Plot
            try:
                self.analysis_report.set_font('Times', 'B', 10)
                self.analysis_report.x = 143
                self.analysis_report.y = 42
                self.analysis_report.cell(50, 4, 'Feature Correlations (Pearson)', 1, align="L")
                self.analysis_report.set_font('Times', '', 8)
                if self.training:
                    self.analysis_report.image(
                        self.experiment_path + '/' + self.datasets[m] + '/exploratory/FeatureCorrelations.png',
                        120, 47, 89, 70)
                    # self.experiment_path + '/' + self.datasets[m] + '/exploratory/FeatureCorrelations.png',
                    # 85, 15, 125, 100)
                    # upper left hand coordinates (x,y),
                    # then image width with hight based on image dimensions (retain original image ratio)
                else:
                    self.analysis_report.image(
                        self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                            m] + '/exploratory/FeatureCorrelations.png', 120, 47, 89, 70)
                    # self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                    #    m] + '/exploratory/FeatureCorrelations.png', 85, 15, 125, 100)
                    # upper left hand coordinates (x,y),
                    # then image width with hight based on image dimensions (retain original image ratio)
            except Exception:
                self.analysis_report.x = 135
                self.analysis_report.y = 60
                self.analysis_report.cell(35, 4, 'No Feature Correlation Plot', 1, align="L")
                pass

            """ #REMOVED FOR REFORMATTING
            if self.training:
                data_summary = pd.read_csv(
                    self.experiment_path + '/' + self.datasets[m] + "/exploratory/DataCounts.csv")
            else:
                data_summary = pd.read_csv(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                        m] + "/exploratory/DataCounts.csv")
            info_ls = []
            for i in range(len(data_summary)):
                info_ls.append(data_summary.iloc[i, 0] + ': ')
                info_ls.append(str(data_summary.iloc[i, 1]))
                info_ls.append('\n')
            self.analysis_report.x = 1
            self.analysis_report.y = 52
            self.analysis_report.set_font('Times', 'B', 8)
            self.analysis_report.multi_cell(w=60, h=4, txt='Dataset Counts Summary:', border=1, align='L')
            self.analysis_report.set_font('Times', '', 8)
            self.analysis_report.multi_cell(w=60, h=4, txt=' ' + list_to_string(info_ls), border=1, align='L')
            """

            # Report Best Algorithms by metric
            if self.training:
                summary_performance = pd.read_csv(
                    self.experiment_path + '/' + self.datasets[m] + "/model_evaluation/Summary_performance_mean.csv")
            else:
                summary_performance = pd.read_csv(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                        m] + "/model_evaluation/Summary_performance_mean.csv")
            summary_performance['ROC AUC'] = summary_performance['ROC AUC'].astype(float)
            highest_roc = summary_performance['ROC AUC'].max()
            algorithm = summary_performance[summary_performance['ROC AUC'] == highest_roc].index.values
            best_alg_roc = summary_performance.iloc[algorithm, 0]

            summary_performance['Balanced Accuracy'] = summary_performance['Balanced Accuracy'].astype(float)
            highest_ba = summary_performance['Balanced Accuracy'].max()
            algorithm = summary_performance[summary_performance['Balanced Accuracy'] == highest_ba].index.values
            best_alg_ba = summary_performance.iloc[algorithm, 0]

            summary_performance['F1 Score'] = summary_performance['F1 Score'].astype(float)
            highest_f1 = summary_performance['F1 Score'].max()
            algorithm = summary_performance[summary_performance['F1 Score'] == highest_f1].index.values
            best_alg_f1 = summary_performance.iloc[algorithm, 0]

            summary_performance['PRC AUC'] = summary_performance['PRC AUC'].astype(float)
            highest_prc = summary_performance['PRC AUC'].max()
            algorithm = summary_performance[summary_performance['PRC AUC'] == highest_prc].index.values
            best_alg_prc = summary_performance.iloc[algorithm, 0]

            summary_performance['PRC APS'] = summary_performance['PRC APS'].astype(float)
            highest_aps = summary_performance['PRC APS'].max()
            algorithm = summary_performance[summary_performance['PRC APS'] == highest_aps].index.values
            best_alg_aps = summary_performance.iloc[algorithm, 0]

            self.analysis_report.x = 1
            self.analysis_report.y = 85
            self.analysis_report.set_font('Times', 'B', 8)
            self.analysis_report.multi_cell(w=80, h=4, txt='Top ML Algorithm Results (Averaged Over CV Runs):',
                                            border=1,
                                            align='L')
            self.analysis_report.set_font('Times', '', 8)

            if len(best_alg_roc.values) > 1:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (ROC_AUC): " + str(
                                                    best_alg_roc.values[0]) + ' (TIE) = ' + str(
                                                    "{:.3f}".format(highest_roc)), border=1, align='L')
            else:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (ROC_AUC): " + str(best_alg_roc.values[0]) + ' = ' + str(
                                                    "{:.3f}".format(highest_roc)), border=1, align='L')

            if len(best_alg_ba.values) > 1:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (Balanced Acc.): " + str(
                                                    best_alg_ba.values[0]) + ' (TIE) = ' + str(
                                                    "{:.3f}".format(highest_ba)), border=1, align='L')
            else:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (Balanced Acc.): " + str(best_alg_ba.values[0]) + ' = ' + str(
                                                    "{:.3f}".format(highest_ba)), border=1, align='L')

            if len(best_alg_f1.values) > 1:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (F1 Score): " + str(
                                                    best_alg_f1.values[0]) + ' (TIE) = ' + str(
                                                    "{:.3f}".format(highest_f1)), border=1, align='L')
            else:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (F1 Score): " + str(best_alg_f1.values[0]) + ' = ' + str(
                                                    "{:.3f}".format(highest_f1)), border=1, align='L')

            if len(best_alg_prc.values) > 1:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (PRC AUC): " + str(
                                                    best_alg_prc.values[0]) + ' (TIE) = ' + str(
                                                    "{:.3f}".format(highest_prc)), border=1, align='L')
            else:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (PRC AUC): " + str(best_alg_prc.values[0]) + ' = ' + str(
                                                    "{:.3f}".format(highest_prc)), border=1, align='L')

            if len(best_alg_aps.values) > 1:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (PRC APS): " + str(
                                                    best_alg_aps.values[0]) + ' (TIE) = ' + str(
                                                    "{:.3f}".format(highest_aps)), border=1, align='L')
            else:
                self.analysis_report.multi_cell(w=80, h=4,
                                                txt="Best (PRC APS): " + str(best_alg_aps.values[0]) + ' = ' + str(
                                                    "{:.3f}".format(highest_aps)), border=1, align='L')

            # self.analysis_report.multi_cell(
            #     w=80, h=4,
            #     txt="Best (ROC_AUC): "
            #         + str(best_alg_roc.values) + ' = '
            #         + str("{:.3f}".format(highest_roc))
            #         + '\n' + "Best (Balanced Acc.): "
            #         + str(best_alg_ba.values)
            #         + ' = ' + str("{:.3f}".format(highest_ba))
            #         + '\n' + "Best (F1 Score): "
            #         + str(best_alg_f1.values) + ' = '
            #         + str("{:.3f}".format(highest_f1))
            #         + '\n' + "Best (PRC AUC): "
            #         + str(best_alg_prc.values) + ' = '
            #         + str("{:.3f}".format(highest_prc))
            #         + '\n' + "Best (PRC APS): "
            #         + str(best_alg_aps.values) + ' = '
            #         + str("{:.3f}".format(highest_aps)), border=1, align='L')

            self.analysis_report.set_font('Times', 'B', 10)
            # ROC
            # -------------------------------
            self.analysis_report.x = 1
            self.analysis_report.y = 112
            self.analysis_report.cell(10, 4, 'ROC', 1, align="L")
            if self.training:
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[m] + '/model_evaluation/Summary_ROC.png', 4, 118,
                    120)
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[
                        m] + '/model_evaluation/metricBoxplots/Compare_ROC AUC.png', 124,
                    118,
                    82, 85)
            else:
                self.analysis_report.image(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                        m] + '/model_evaluation/Summary_ROC.png',
                    4, 118, 120)
                self.analysis_report.image(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                        m] + '/model_evaluation/metricBoxplots/Compare_ROC AUC.png', 124, 118, 82, 85)

            # PRC-------------------------------
            self.analysis_report.x = 1
            self.analysis_report.y = 200
            self.analysis_report.cell(10, 4, 'PRC', 1, align="L")
            if self.training:
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[m] + '/model_evaluation/Summary_PRC.png', 4, 206,
                    133)  # wider to account for more text
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[
                        m] + '/model_evaluation/metricBoxplots/Compare_PRC AUC.png', 138,
                    205,
                    68, 80)
            else:
                self.analysis_report.image(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                        m] + '/model_evaluation/Summary_PRC.png',
                    4, 206, 133)  # wider to account for more text
                self.analysis_report.image(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                        m] + '/model_evaluation/metricBoxplots/Compare_PRC AUC.png', 138, 205, 68, 80)
            self.footer()

        # NEXT PAGE(S) - Average Model Prediction Statistics
        # --------------------------------------------------------------------------------------
        logging.info("Publishing Average Model Prediction Statistics")
        result_limit = 5  # Limits to this many dataset results per page
        dataset_count = len(self.datasets)
        # Determine number of pages needed for univariate results
        page_count = dataset_count / float(result_limit)
        page_count = math.ceil(page_count)  # rounds up to next full integer
        self.analysis_report.set_fill_color(200)
        for page in range(0, page_count):  # generate each page
            self.pub_model_mean_stats(page, result_limit, page_count)

        # NEXT PAGE(S) - Median Model Prediction Statistics
        # --------------------------------------------------------------------------------------
        logging.info("Publishing Median Model Prediction Statistics")
        result_limit = 5  # Limits to this many dataset results per page
        dataset_count = len(self.datasets)
        # Determine number of pages needed for univariate results
        page_count = dataset_count / float(result_limit)
        page_count = math.ceil(page_count)  # rounds up to next full integer
        self.analysis_report.set_fill_color(200)
        for page in range(0, page_count):  # generate each page
            self.pub_model_median_stats(page, result_limit, page_count)

        # NEXT PAGE(S) - ML Dataset Feature Importance Summary
        # ----------------------------------------------------------------
        if self.training:
            logging.info("Publishing Feature Importance Summaries")
            for k in range(len(self.datasets)):
                self.analysis_report.add_page()
                self.analysis_report.set_font('Times', 'B', 12)
                self.analysis_report.cell(w=0, h=8,
                                          txt="Feature Importance Summary:  D" + str(k + 1) + ' = ' + self.datasets[k],
                                          border=1, align="L", ln=2)
                self.analysis_report.set_font(family='times', size=9)
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[
                        k] + '/feature_selection/mutual_information/TopAverageScores.png',
                    5,
                    12, 100, 135)  # Images adjusted to fit a width of 100 and length of 135
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[k] + '/feature_selection/multisurf/TopAverageScores.png',
                    105, 12,
                    100,
                    135)
                self.analysis_report.x = 0
                self.analysis_report.y = 150
                self.analysis_report.cell(0, 8,
                                          "Composite Feature Importance Plot (Normalized and Performance Weighted)", 1,
                                          align="L")
                self.analysis_report.image(
                    self.experiment_path + '/' + self.datasets[
                        k] + '/model_evaluation/feature_importance/Compare_FI_Norm_Weight.png',
                    1, 159, 208, 125)  # 130 added
                self.footer()

        # NEXT PAGE - Create Dataset Boxplot Comparison Page
        # ---------------------------------------
        if self.training:
            logging.info("Publishing Dataset Comparison Boxplots")
            self.analysis_report.add_page()
            self.analysis_report.set_font('Times', 'B', 12)
            self.analysis_report.cell(w=0, h=8, txt="Compare ML Performance Across Datasets", border=1, align="L",
                                      ln=2)
            self.analysis_report.set_font(family='times', size=9)
            if len(self.datasets) > 1:
                self.analysis_report.image(
                    self.experiment_path + '/DatasetComparisons/dataCompBoxplots/' + 'DataCompareAllModels_ROC AUC.png',
                    1,
                    12,
                    208, 130)  # Images adjusted to fit a width of 100 and length of 135
                self.analysis_report.image(
                    self.experiment_path + '/DatasetComparisons/dataCompBoxplots/' + 'DataCompareAllModels_PRC AUC.png',
                    1,
                    150,
                    208, 130)  # Images adjusted to fit a width of 100 and length of 135
            self.footer()

        # NEXT PAGE(S) -Create Best Kruskall Wallis Dataset Comparison Page
        # ---------------------------------------
        if self.training:
            logging.info("Publishing Statistical Analysis")
            self.analysis_report.add_page(orientation='P')
            self.analysis_report.set_margins(left=1, top=10, right=1, )

            d = []
            for i in range(len(self.datasets)):
                d.append('Data ' + str(i + 1) + '= ' + self.datasets[i])
                d.append('\n')

            self.analysis_report.set_font('Times', 'B', 12)
            self.analysis_report.cell(w=0, h=8,
                                      txt='Using Best Performing Algorithms (Kruskall Wallis Compare Datasets)',
                                      border=1, align="L", ln=2)
            self.analysis_report.set_font(family='times', size=7)

            # Dataset list Key
            list_datasets = ''
            i = 1
            for each in self.datasets:
                list_datasets = list_datasets + ('D' + str(i) + ' = ' + str(each) + '\n')
                i += 1
            self.analysis_report.x = 5
            self.analysis_report.y = 14
            self.analysis_report.multi_cell(w=0, h=4, txt='Datasets: ' + '\n' + list_datasets, border=1, align='L')
            self.analysis_report.y += 5

            success = False
            kruskal_wallis_datasets = None
            try:
                # Kruskal Wallis Table
                # A table can take at most 4 self.datasets to fit comfortably with these settings
                kruskal_wallis_datasets = pd.read_csv(self.experiment_path + '/DatasetComparisons/' +
                                                      'BestCompare_KruskalWallis.csv', sep=',', index_col=0)
                kruskal_wallis_datasets = kruskal_wallis_datasets.round(4)
                success = True
            except Exception:
                pass

            if success:
                # Process
                # for i in range(len(self.datasets)):
                #    kruskal_wallis_datasets = kruskal_wallis_datasets.drop('Std_D'+str(i+1),1)
                kruskal_wallis_datasets = kruskal_wallis_datasets.drop('Statistic', axis=1)
                kruskal_wallis_datasets = kruskal_wallis_datasets.drop('Sig(*)', axis=1)

                # Format
                kruskal_wallis_datasets.reset_index(inplace=True)
                temp_df = pd.concat([kruskal_wallis_datasets.columns.to_frame().T, kruskal_wallis_datasets])
                temp_df.iloc[0, 0] = 'Metrics'
                kruskal_wallis_datasets = temp_df
                kruskal_wallis_datasets.columns = range(len(kruskal_wallis_datasets.columns))
                # epw = 208  # Amount of Space (width) Available
                th = self.analysis_report.font_size
                # col_width = epw/float(10) #maximum column width
                col_width_list = [23, 14, 30, 14, 30, 14, 30, 14]

                if len(self.datasets) <= 3:  # 4
                    col_count = 0
                    kruskal_wallis_datasets = kruskal_wallis_datasets.to_numpy()
                    for row in kruskal_wallis_datasets:
                        for datum in row:
                            self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1)
                            col_count += 1
                        col_count = 0
                        self.analysis_report.ln(th)  # critical
                else:
                    # Print next 3 self.datasets
                    col_count = 0
                    table1 = kruskal_wallis_datasets.iloc[:, :8]  # 10
                    table1 = table1.to_numpy()
                    for row in table1:
                        for datum in row:
                            self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1)
                            col_count += 1
                        col_count = 0
                        self.analysis_report.ln(th)  # critical
                    self.analysis_report.y += 5

                    col_count = 0
                    table1 = kruskal_wallis_datasets.iloc[:, 8:14]  # 10:18
                    met = kruskal_wallis_datasets.iloc[:, 0]
                    met2 = kruskal_wallis_datasets.iloc[:, 1]
                    table1 = pd.concat([met, met2, table1], axis=1)
                    table1 = table1.to_numpy()
                    for row in table1:
                        for datum in row:
                            self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1)
                            col_count += 1
                        col_count = 0
                        self.analysis_report.ln(th)  # critical
                    self.analysis_report.y += 5

                    if len(self.datasets) > 6:  # 8
                        col_count = 0
                        table1 = kruskal_wallis_datasets.iloc[:, 14:20]  # 18:26
                        met = kruskal_wallis_datasets.iloc[:, 0]
                        met2 = kruskal_wallis_datasets.iloc[:, 1]
                        table1 = pd.concat([met, met2, table1], axis=1)
                        table1 = table1.to_numpy()
                        for row in table1:
                            for datum in row:
                                self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1)
                                col_count += 1
                            col_count = 0
                            self.analysis_report.ln(th)  # critical
                        self.analysis_report.y += 5

                    if len(self.datasets) > 9:
                        table1 = kruskal_wallis_datasets.iloc[:, 20:26]
                        met = kruskal_wallis_datasets.iloc[:, 0]
                        met2 = kruskal_wallis_datasets.iloc[:, 1]
                        table1 = pd.concat([met, met2, table1], axis=1)
                        table1 = table1.to_numpy()
                        for row in table1:
                            for datum in row:
                                self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1)
                                col_count += 1
                            col_count = 0
                            self.analysis_report.ln(th)  # critical
                        self.analysis_report.y += 5

                    if len(self.datasets) > 12:
                        self.analysis_report.x = 0
                        self.analysis_report.y = 260
                        self.analysis_report.cell(0, 4, 'Warning: Additional dataset results could not be displayed', 1,
                                                  align="C")

            self.footer()

        # LAST PAGE - Create Runtime Summary Page---------------------------------------
        if self.training:
            logging.info("Publishing Runtime Summary")
            result_limit = 6  # Limits to this many dataset results per page
            dataset_count = len(self.datasets)
            # Determine number of pages needed for univariate results
            page_count = dataset_count / float(result_limit)
            page_count = math.ceil(page_count)  # rounds up to next full integer
            for page in range(0, page_count):  # generate each page
                self.pub_runtime(page, result_limit, page_count)

        # Output The PDF Object
        try:
            if self.training:
                file_name = str(self.experiment_name) + '_ML_Pipeline_Report.pdf'
                self.analysis_report.output(self.experiment_path + '/' + file_name)
                # Print phase completion
                logging.info("Phase 8 complete")
                try:
                    job_file = open(self.experiment_path + '/jobsCompleted/job_data_pdf_training.txt', 'w')
                    job_file.write('complete')
                    job_file.close()
                except Exception:
                    pass
            else:
                file_name = str(self.experiment_name) + '_ML_Pipeline_Apply_Report.pdf'
                self.analysis_report.output(
                    self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[M] + '/' + file_name)
                # Print phase completion
                logging.info("Phase 10 complete")
                try:
                    job_file = open(self.experiment_path + '/jobsCompleted/job_data_pdf_apply_' + str(
                        self.train_name) + '.txt',
                                    'w')
                    job_file.write('complete')
                    job_file.close()
                except Exception:
                    pass
        except Exception:
            logging.info('Pdf Output Failed')

    def pub_univariate(self, page, result_limit, page_count):
        """ Generates single page of univariate analysis results. Automatically moves to another page when runs out of
        space. Maximum of 4 dataset results to a page."""
        dataset_count = len(self.datasets)
        data_start = page * result_limit
        count_limit = (page * result_limit) + result_limit
        self.analysis_report.add_page(orientation='P')
        self.analysis_report.set_font('Times', 'B', 12)
        if page_count > 1:
            self.analysis_report.cell(w=180, h=8,
                                      txt='Univariate Analysis of Each Dataset (Top 10 Features for Each): Page ' + str(
                                          page + 1),
                                      border=1, align='L', ln=2)
        else:
            self.analysis_report.cell(w=180, h=8, txt='Univariate Analysis of Each Dataset (Top 10 Features for Each)',
                                      border=1,
                                      align='L', ln=2)
        try:
            # Try loop added to deal with versions specific change to using mannwhitneyu in scipy and
            # avoid STREAMLINE crash in those circumstances.
            for n in range(data_start, dataset_count):
                if n >= count_limit:  # Stops generating page when dataset count limit reached
                    break
                self.analysis_report.y += 2
                sig_df = pd.read_csv(
                    self.experiment_path + '/' + self.datasets[
                        n] + '/exploratory/univariate_analyses/Univariate_Significance.csv')
                sig_ls = []
                sig_df = sig_df.nsmallest(10, ['p-value'])

                self.analysis_report.set_font('Times', 'B', 10)
                self.analysis_report.multi_cell(w=160, h=6, txt='D' + str(n + 1) + ' = ' + self.datasets[n], border=0,
                                                align='L')

                # for i in range(len(sig_df)):
                #     sig_ls.append(sig_df.iloc[i, 0] + '\t\t\t: ')
                #     sig_ls.append(str(sig_df.iloc[i, 1]))
                #     sig_ls.append('\t\t\t' + '(' + sig_df.iloc[i, 3] + ',' + str(sig_df.iloc[i, 2]) + ')' + '\n')
                # self.analysis_report.set_font('Times', 'B', 10)
                # self.analysis_report.multi_cell(w=180, h=4, txt='D' + str(n + 1) + ' = ' + self.datasets[n], border=1,
                #                                 align='L')
                # self.analysis_report.y += 1  # Space below section header
                # self.analysis_report.set_font('Times', 'B', 8)
                # self.analysis_report.multi_cell(w=180, h=4,
                # txt='Feature: \t\t\t P-Value \t\t\t (Test, test statistics)', border=1,
                #                                 align='L')
                # self.analysis_report.set_font('Times', '', 8)
                # self.analysis_report.multi_cell(w=180, h=4, txt=' ' + list_to_string(sig_ls), border=1, align='L')

                self.analysis_report.set_font('Times', '', 8)
                # sig_df = sig_df.round(3)
                # Format
                sig_df.reset_index(inplace=True)
                sig_df = pd.concat([sig_df.columns.to_frame().T, sig_df])
                sig_df.columns = range(len(sig_df.columns))
                th = self.analysis_report.font_size
                col_width_list = [40, 40, 40, 40, 40, 20]
                table1 = sig_df.iloc[:, :]
                table1 = table1.to_numpy()

                # Print table header first
                row_count = 0
                col_count = 0

                for row in table1:  # each row
                    for datum in row:
                        entry_list = str(datum).split(' ')
                        try:
                            if col_count == 0:
                                pass
                            else:
                                self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1,
                                                          align="C")
                        except Exception:
                            self.analysis_report.cell(col_width_list[col_count], th, ' ', border=1, align="C")
                        col_count += 1
                    self.analysis_report.ln(th)  # critical
                    col_count = 0
                    row_count += 1

        except Exception as e:
            self.analysis_report.x = 5
            self.analysis_report.y = 40
            self.analysis_report.cell(180, 4,
                                      # 'WARNING: Univariate analysis failed from scipy package error. To fix: pip '
                                      # 'install --upgrade scipy',
                                      str(e),
                                      1, align="L")
        self.footer()

    def pub_model_mean_stats(self, page, result_limit, page_count):
        dataset_count = len(self.datasets)
        data_start = page * result_limit
        count_limit = (page * result_limit) + result_limit
        # Create PDF and Set Options
        self.analysis_report.set_margins(left=1, top=1, right=1, )
        self.analysis_report.add_page()
        self.analysis_report.set_font('Times', 'B', 12)
        if page_count > 1:
            self.analysis_report.cell(w=0, h=8,
                                      txt='Average Model Prediction Statistics (Rounded to 3 Decimal Points): Page '
                                          + str(page + 1), border=1, align='L', ln=2)
        else:
            self.analysis_report.cell(w=0, h=8, txt='Average Model Prediction Statistics (Rounded to 3 Decimal Points)',
                                      border=1,
                                      align='L', ln=2)
        for n in range(data_start, dataset_count):
            if n >= count_limit:
                # Stops generating page when dataset count limit reached
                break
            self.analysis_report.y += 4
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.multi_cell(w=0, h=4, txt='D' + str(n + 1) + ' = ' + self.datasets[n], border=1,
                                            align='L')
            self.analysis_report.y += 1  # Space below section header
            self.analysis_report.set_font('Times', '', 7)
            if self.training:
                stats_ds = pd.read_csv(
                    self.experiment_path + '/' + str(
                        self.datasets[n]) + '/model_evaluation/Summary_performance_mean.csv',
                    sep=',',
                    index_col=0)
            else:
                stats_ds = pd.read_csv(self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                    n] + '/model_evaluation/Summary_performance_mean.csv', sep=',', index_col=0)
            # Make list of top values for each metric
            metric_name_list = ['Balanced Accuracy', 'Accuracy', 'F1 Score', 'Sensitivity (Recall)', 'Specificity',
                                'Precision (PPV)', 'TP', 'TN', 'FP', 'FN', 'NPV', 'LR+', 'LR-', 'ROC AUC', 'PRC AUC',
                                'PRC APS']
            best_metric_list = []
            if self.training:
                ds2 = pd.read_csv(
                    self.experiment_path + '/' + self.datasets[n] + "/model_evaluation/Summary_performance_mean.csv")
            else:
                ds2 = pd.read_csv(self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                    n] + '/model_evaluation/Summary_performance_mean.csv')

            self.format_fn(stats_ds, best_metric_list, metric_name_list, ds2)

        self.footer()

    def pub_model_median_stats(self, page, result_limit, page_count):
        dataset_count = len(self.datasets)
        data_start = page * result_limit
        count_limit = (page * result_limit) + result_limit
        # Create PDF and Set Options
        self.analysis_report.set_margins(left=1, top=1, right=1, )
        self.analysis_report.add_page()
        self.analysis_report.set_font('Times', 'B', 12)
        if page_count > 1:
            self.analysis_report.cell(w=0, h=8,
                                      txt='Median Model Prediction Statistics (Rounded to 3 Decimal Points): Page '
                                          + str(page + 1),
                                      border=1, align='L', ln=2)
        else:
            self.analysis_report.cell(w=0, h=8, txt='Median Model Prediction Statistics (Rounded to 3 Decimal Points)',
                                      border=1,
                                      align='L', ln=2)
        for n in range(data_start, dataset_count):
            if n >= count_limit:  # Stops generating page when dataset count limit reached
                break
            self.analysis_report.y += 4
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.multi_cell(w=0, h=4, txt='D' + str(n + 1) + ' = ' + self.datasets[n], border=1,
                                            align='L')
            self.analysis_report.y += 1  # Space below section header
            self.analysis_report.set_font('Times', '', 7)
            if self.training:
                stats_ds = pd.read_csv(
                    self.experiment_path + '/' + str(
                        self.datasets[n]) + '/model_evaluation/Summary_performance_median.csv',
                    sep=',',
                    index_col=0)
            else:
                stats_ds = pd.read_csv(self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                    n] + '/model_evaluation/Summary_performance_median.csv', sep=',', index_col=0)
            # Make list of top values for each metric
            metric_name_list = ['Balanced Accuracy', 'Accuracy', 'F1 Score', 'Sensitivity (Recall)', 'Specificity',
                                'Precision (PPV)', 'TP', 'TN', 'FP', 'FN', 'NPV', 'LR+', 'LR-', 'ROC AUC', 'PRC AUC',
                                'PRC APS']
            best_metric_list = []
            if self.training:
                ds2 = pd.read_csv(
                    self.experiment_path + '/' + self.datasets[n] + "/model_evaluation/Summary_performance_median.csv")
            else:
                ds2 = pd.read_csv(self.experiment_path + '/' + self.train_name + '/applymodel/' + self.datasets[
                    n] + '/model_evaluation/Summary_performance_median.csv')
            self.format_fn(stats_ds, best_metric_list, metric_name_list, ds2)
        self.footer()

    def pub_runtime(self, page, result_limit, page_count):
        """
        Generates single page of runtime analysis results. Automatically moves to another page when runs out of
        space. Maximum of 4 dataset results to a page.
        """
        col_width = 40  # maximum column width
        dataset_count = len(self.datasets)
        data_start = page * result_limit
        count_limit = (page * result_limit) + result_limit
        self.analysis_report.add_page(orientation='P')
        self.analysis_report.set_font('Times', 'B', 12)
        if page_count > 1:
            self.analysis_report.cell(w=0, h=8, txt='Pipeline Runtime Summary: Page ' + str(page + 1), border=1,
                                      align='L',
                                      ln=2)
        else:
            self.analysis_report.cell(w=0, h=8, txt='Pipeline Runtime Summary', border=1, align='L', ln=2)
        self.analysis_report.set_font('Times', '', 8)
        th = self.analysis_report.font_size
        self.analysis_report.y += 2
        left = True
        for n in range(data_start, dataset_count):
            if n >= count_limit:  # Stops generating page when dataset count limit reached
                break
            last_y = self.analysis_report.y
            last_x = self.analysis_report.x
            if left:
                self.analysis_report.x = 1
            time_df = pd.read_csv(self.experiment_path + '/' + self.datasets[n] + '/runtimes.csv')
            time_df.iloc[:, 1] = time_df.iloc[:, 1].round(2)
            time_df = pd.concat([time_df.columns.to_frame().T, time_df])
            time_df = time_df.to_numpy()
            self.analysis_report.set_font('Times', 'B', 10)
            self.analysis_report.cell(col_width * 2, 4, str(self.datasets[n]), 1, align="L")
            self.analysis_report.y += 5
            self.analysis_report.x = last_x
            self.analysis_report.set_font('Times', '', 7)
            for row in time_df:
                for datum in row:
                    self.analysis_report.cell(col_width, th, str(datum), border=1)
                self.analysis_report.ln(th)  # critical
                self.analysis_report.x = last_x

            if left:
                self.analysis_report.x = (col_width * 2) + 2
                self.analysis_report.y = last_y
                left = False
            else:
                self.analysis_report.x = 1
                self.analysis_report.y = last_y + 75
                left = True
        self.footer()

    def format_fn(self, stats_ds, best_metric_list, metric_name_list, ds2):
        low_val_better = ['FP', 'FN', 'LR-']
        for metric in metric_name_list:
            if metric in low_val_better:
                ds2[metric] = ds2[metric].astype(float).round(3)
                metric_best = ds2[metric].min()
            else:
                ds2[metric] = ds2[metric].astype(float).round(3)
                metric_best = ds2[metric].max()
            best_metric_list.append(metric_best)

        stats_ds = stats_ds.round(3)
        # Format
        stats_ds.reset_index(inplace=True)
        stats_ds = pd.concat([stats_ds.columns.to_frame().T, stats_ds])
        stats_ds.columns = range(len(stats_ds.columns))
        th = self.analysis_report.font_size
        col_width_list = [32, 11, 11, 8, 12, 12, 10, 15, 15, 15, 15, 8, 9, 9, 8, 8, 8]
        table1 = stats_ds.iloc[:, :18]
        table1 = table1.to_numpy()

        # Print table header first
        row_count = 0
        col_count = 0

        for row in table1:  # each row
            if row_count == 0:
                # Print first row
                for datum in row:
                    if col_count == 0:
                        self.analysis_report.cell(col_width_list[col_count], th, 'ML Algorithm', border=0, align="C")
                    else:
                        entry_list = str(datum).split(' ')
                        self.analysis_report.cell(col_width_list[col_count], th, entry_list[0], border=0, align="C")
                    col_count += 1
                self.analysis_report.ln(th)  # critical
                col_count = 0
                # Print second row
                for datum in row:
                    entry_list = str(datum).split(' ')
                    try:
                        self.analysis_report.cell(col_width_list[col_count], th, entry_list[1], border=0, align="C")
                    except Exception:
                        self.analysis_report.cell(col_width_list[col_count], th, ' ', border=0, align="C")
                    col_count += 1
                self.analysis_report.ln(th)  # critical
                col_count = 0
            else:  # Print table contents
                for datum in row:  # each column
                    if col_count > 0 and float(datum) == float(best_metric_list[col_count - 1]):
                        self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1, align="L",
                                                  fill=True)
                    else:
                        self.analysis_report.cell(col_width_list[col_count], th, str(datum), border=1, align="L")
                    col_count += 1
                self.analysis_report.ln(th)  # critical
                col_count = 0
            row_count += 1

    def footer(self):
        self.analysis_report.set_auto_page_break(auto=False, margin=3)
        self.analysis_report.set_y(285)
        self.analysis_report.set_font('Times', 'I', 7)
        self.analysis_report.cell(0, 7,
                                  'Generated with STREAMLINE (' + version
                                  + '): (https://github.com/UrbsLab/STREAMLINE)', 0,
                                  0, 'C')
        self.analysis_report.set_font(family='times', size=9)


def list_to_string(s):
    """Convert a list of string to string"""
    str1 = " "
    return str1.join(s)


def ngi(list1, n):
    """Find N the greatest integers within a list"""
    final_list = []
    for i in range(0, n):
        max1 = 0
        for j in range(len(list1)):
            if list1[j] > max1:
                max1 = list1[j]
        list1.remove(max1)
        final_list.append(max1)
