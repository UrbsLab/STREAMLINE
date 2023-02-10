import os
import time
import logging
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import kruskal, wilcoxon, mannwhitneyu
from streamline.utils.job import Job
from streamline.modeling.utils import ABBREVIATION, COLORS, is_supported_model
from streamline.modeling.utils import SUPPORTED_MODELS
import seaborn as sns
sns.set_theme()


class CompareJob(Job):
    """
    This 'Job' script is called by DataCompareMain.py which runs non-parametric statistical analysis
    comparing ML algorithm performance between all target datasets included in the original Phase 1 data folder,
    for each evaluation metric.
    Also compares the best overall model for each target dataset, for each evaluation metric.
    This runs once for the entire pipeline analysis.
    """
    def __init__(self, output_path=None, experiment_name=None, experiment_path=None, algorithms=None,
                 exclude=("XCS", "eLCS"),
                 class_label="Class", instance_label=None, sig_cutoff=0.05, show_plots=False):
        super().__init__()
        assert (output_path is not None and experiment_name is not None) or (experiment_path is not None)
        if output_path is not None and experiment_name is not None:
            self.output_path = output_path
            self.experiment_name = experiment_name
            self.experiment_path = self.output_path + '/' + self.experiment_name
        else:
            self.experiment_path = experiment_path
            self.experiment_name = self.experiment_path.split('/')[-1]
            self.output_path = self.experiment_path.split('/')[-2]

        datasets = os.listdir(self.experiment_path)
        remove_list = ['metadata.pickle', 'metadata.csv', 'algInfo.pickle',
                       'jobsCompleted', 'logs', 'jobs', 'DatasetComparisons', 'UsefulNotebooks',
                       self.experiment_name + '_ML_Pipeline_Report.pdf']
        for text in remove_list:
            if text in datasets:
                datasets.remove(text)
        # ensures consistent ordering of datasets and assignment of temporary identifier
        self.datasets = sorted(datasets)

        dataset_directory_paths = []
        for dataset in self.datasets:
            full_path = self.experiment_path + "/" + dataset
            dataset_directory_paths.append(full_path)

        self.dataset_directory_paths = dataset_directory_paths

        self.class_label = class_label
        self.instance_label = instance_label
        self.sig_cutoff = sig_cutoff

        if algorithms is None:
            self.algorithms = SUPPORTED_MODELS
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

        self.show_plots = show_plots
        self.abbrev = dict((k, ABBREVIATION[k]) for k in self.algorithms if k in ABBREVIATION)
        self.colors = dict((k, COLORS[k]) for k in self.algorithms if k in COLORS)
        self.metrics = None

    def run(self):
        self.job_start_time = time.time()  # for tracking phase runtime

        data = pd.read_csv(self.dataset_directory_paths[0] + '/model_evaluation/Summary_performance_mean.csv', sep=',')
        self.metrics = data.columns.values.tolist()[1:]

        # Create directory to store dataset statistical comparisons
        if not os.path.exists(self.experiment_path + '/DatasetComparisons'):
            os.mkdir(self.experiment_path + '/DatasetComparisons')

        logging.info('Running Statistical Significance Comparisons Between Multiple Datasets...')

        self.kruscall_wallis()

        self.mann_whitney_u()

        self.wilcoxon_rank()

        global_data = self.best_kruscall_wallis()

        self.best_mann_whitney_u(global_data)

        self.best_wilcoxon_rank(global_data)

        logging.info('Generate Boxplots Comparing Dataset Performance...')
        # Generate boxplots comparing average algorithm performance
        # (for a given metric) across all dataset comparisons
        self.data_compare_bp_all()

        # Generate boxplots comparing a specific algorithm's CV performance (
        # for AUC_ROC or AUC_PRC) across all dataset comparisons
        self.data_compare_bp()
        # Print phase completion
        logging.info("Phase 7 complete")
        job_file = open(self.experiment_path + '/jobsCompleted/job_data_compare' + '.txt', 'w')
        job_file.write('complete')
        job_file.close()

    def kruscall_wallis(self):
        """
        For each algorithm apply non-parametric Kruskal Wallis one-way ANOVA on ranks.
        Determines if there is a statistically significant difference in performance
        between original target datasets across CV runs.
        Completed for each standard metric separately.
        """

        label = ['Statistic', 'P-Value', 'Sig(*)']
        for i in range(1, len(self.datasets) + 1):
            label.append('Median_D' + str(i))

        for algorithm in self.algorithms:
            kruskal_summary = pd.DataFrame(index=self.metrics, columns=label)
            for metric in self.metrics:
                temp_array = []
                med_list = []
                for dataset_path in self.dataset_directory_paths:
                    filename = dataset_path + '/model_evaluation/' + self.abbrev[algorithm] + '_performance.csv'
                    td = pd.read_csv(filename)
                    temp_array.append(td[metric])
                    med_list.append(td[metric].median())
                try:  # Run kruskal Wallis
                    result = kruskal(*temp_array)
                except Exception:
                    result = ['NA', 1]
                kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
                kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
                if result[1] < self.sig_cutoff:
                    kruskal_summary.at[metric, 'Sig(*)'] = str('*')
                else:
                    kruskal_summary.at[metric, 'Sig(*)'] = str('')
                for j in range(len(med_list)):
                    kruskal_summary.at[metric, 'Median_D' + str(j + 1)] = str(round(med_list[j], 6))
            # Export analysis summary to .csv file
            kruskal_summary.to_csv(self.experiment_path + '/DatasetComparisons/KruskalWallis_' + algorithm + '.csv')

    def wilcoxon_rank(self):
        """
        For each algorithm, apply non-parametric Wilcoxon Rank Sum (pairwise comparisons).
        This tests individual algorithm pairs of original target datasets (for each metric)
        to determine if there is a statistically significant difference in performance across CV runs.
        Test statistic will be zero if all scores from one set are
        larger than the other.
        """

        label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
        for i in range(1, 3):
            label.append('Median_Data' + str(i))

        for algorithm in self.algorithms:
            master_list = self.inter_set_fn(wilcoxon, algorithm)
            # Export test results
            df = pd.DataFrame(master_list)
            df.columns = label
            df.to_csv(self.experiment_path + '/DatasetComparisons/WilcoxonRank_' + algorithm + '.csv', index=False)

    def mann_whitney_u(self):
        """
        For each algorithm, apply non-parametric Mann Whitney U-test (pairwise comparisons).
        Mann Whitney tests dataset pairs (for each metric)
        to determine if there is a statistically significant difference in performance across CV runs.
        Test statistic will be zero if all scores from one set are
        larger than the other.
        """

        label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
        for i in range(1, 3):
            label.append('Median_Data' + str(i))
        for algorithm in self.algorithms:
            # Export test results
            master_list = self.inter_set_fn(mannwhitneyu, algorithm)
            df = pd.DataFrame(master_list)
            df.columns = label
            df.to_csv(self.experiment_path + '/DatasetComparisons/MannWhitney_' + algorithm + '.csv', index=False)

    def best_kruscall_wallis(self):
        """
        For best performing algorithm on a given metric and dataset, apply non-parametric
        Kruskal Wallis one-way ANOVA on ranks.
        Determines if there is a statistically significant difference in performance
        between original target datasets across CV runs
        on best algorithm for given metric.
        """

        label = ['Statistic', 'P-Value', 'Sig(*)']
        for i in range(1, len(self.datasets) + 1):
            label.append('Best_Alg_D' + str(i))
            label.append('Median_D' + str(i))

        kruskal_summary = pd.DataFrame(index=self.metrics, columns=label)
        global_data = []

        for metric in self.metrics:
            best_list = []
            best_data = []
            for dataset_path in self.dataset_directory_paths:
                alg_med = []
                alg_data = []
                for algorithm in self.algorithms:
                    filename = dataset_path + '/model_evaluation/' + self.abbrev[algorithm] + '_performance.csv'
                    td = pd.read_csv(filename)
                    alg_med.append(td[metric].median())
                    alg_data.append(td[metric])
                # Find the best algorithm for given metric based on average
                best_med = max(alg_med)
                best_index = alg_med.index(best_med)
                best_alg = self.algorithms[best_index]
                best_data.append(alg_data[best_index])
                best_list.append([best_alg, best_med])
            global_data.append([best_data, best_list])
            try:
                result = kruskal(*best_data)
                kruskal_summary.at[metric, 'Statistic'] = str(round(result[0], 6))
                kruskal_summary.at[metric, 'P-Value'] = str(round(result[1], 6))
                if result[1] < self.sig_cutoff:
                    kruskal_summary.at[metric, 'Sig(*)'] = str('*')
                else:
                    kruskal_summary.at[metric, 'Sig(*)'] = str('')
            except ValueError:
                kruskal_summary.at[metric, 'Statistic'] = str(round(np.nan, 6))
                kruskal_summary.at[metric, 'P-Value'] = str(round(np.nan, 6))
                kruskal_summary.at[metric, 'Sig(*)'] = str('')
            for j in range(len(best_list)):
                kruskal_summary.at[metric, 'Best_Alg_D' + str(j + 1)] = str(best_list[j][0])
                kruskal_summary.at[metric, 'Median_D' + str(j + 1)] = str(round(best_list[j][1], 6))
        # Export analysis summary to .csv file
        kruskal_summary.to_csv(self.experiment_path + '/DatasetComparisons/BestCompare_KruskalWallis.csv')
        return global_data

    def best_mann_whitney_u(self, global_data):
        """
        For best performing algorithm on a given metric and dataset,
        apply non-parametric Mann Whitney U-test (pairwise comparisons).
        Mann Whitney tests dataset pairs (for each metric)
        to determine if there is a statistically significant difference
        in performance across CV runs. Test statistic will be zero if all scores from one set are
        larger than the other.
        """
        df = self.inter_set_best_fn(mannwhitneyu, global_data)
        df.to_csv(self.experiment_path + '/DatasetComparisons/BestCompare_MannWhitney.csv', index=False)

    def best_wilcoxon_rank(self, global_data):
        """
        For best performing algorithm on a given metric and dataset, apply
        non-parametric Mann Whitney U-test (pairwise comparisons).
        Mann Whitney tests dataset pairs (for each metric)
        to determine if there is a statistically significant difference in
        performance across CV runs. Test statistic will be zero if all scores from one set are
        larger than the other.
        """
        df = self.inter_set_best_fn(wilcoxon, global_data)
        df.to_csv(self.experiment_path + '/DatasetComparisons/BestCompare_WilcoxonRank.csv', index=False)

    def data_compare_bp_all(self):
        """
        Generate a boxplot comparing algorithm performance (CV average of each target metric)
        across all target datasets to be compared.
        """

        if not os.path.exists(self.experiment_path + '/DatasetComparisons/dataCompBoxplots'):
            os.mkdir(self.experiment_path + '/DatasetComparisons/dataCompBoxplots')

        # One boxplot generated for each available metric
        for metric in self.metrics:
            df = pd.DataFrame()
            data_name_list = []
            alg_values_dict = {}
            # Dictionary of all algorithms run that will each have a list of respective mean metric value
            for algorithm in self.algorithms:
                # Used to generate algorithm lines on top of boxplot
                alg_values_dict[algorithm] = []
            # For each target dataset
            for each in self.dataset_directory_paths:
                data_name_list.append(each.split('/')[-1])
                data = pd.read_csv(each + '/model_evaluation/Summary_performance_mean.csv', sep=',', index_col=0)
                rownames = data.index.values  # makes a list of algorithm names from file
                rownames = list(rownames)
                # Grab data in metric column
                col = data[metric]  # Dataframe of average target metric values for each algorithm
                col_list = data[metric].tolist()  # List of average target metric values for each algorithm
                for j in range(len(rownames)):  # For each algorithm
                    alg_values_dict[rownames[j]].append(col_list[j])
                # Create dataframe of average target metric where columns are datasets, and rows are algorithms
                df = pd.concat([df, col], axis=1)
            df.columns = data_name_list
            # Generate boxplot (with legend for each box) ---------------------------------------
            # Plot boxplots
            df.boxplot(column=data_name_list, rot=90)
            # Plot lines for each algorithm (to illustrate algorithm performance trajectories between datasets)
            for i in range(len(self.algorithms)):
                plt.plot(np.arange(len(self.dataset_directory_paths)) + 1, alg_values_dict[self.algorithms[i]],
                         color=self.colors[self.algorithms[i]], label=self.algorithms[i])
            # Specify plot labels
            plt.ylabel(str(metric))
            plt.xlabel('Dataset')
            plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
            # Export and/or show plot
            plt.savefig(
                self.experiment_path + '/DatasetComparisons/dataCompBoxplots/DataCompareAllModels_' + metric + '.png',
                bbox_inches="tight")
            if self.show_plots:
                plt.show()
            else:
                plt.close('all')

    def data_compare_bp(self):
        """
        Generate a boxplot comparing average algorithm performance (for a given target metric)
        across all target datasets to be compared.
        """
        metric_list = ['ROC AUC', 'PRC AUC']  # Hard coded
        if not os.path.exists(self.experiment_path + '/DatasetComparisons/dataCompBoxplots'):
            os.mkdir(self.experiment_path + '/DatasetComparisons/dataCompBoxplots')
        for algorithm in self.algorithms:
            for metric in metric_list:
                df = pd.DataFrame()
                data_name_list = []
                for each in self.dataset_directory_paths:
                    data_name_list.append(each.split('/')[-1])
                    data = pd.read_csv(each + '/model_evaluation/' + self.abbrev[algorithm] + '_performance.csv',
                                       sep=',')
                    # Grab data in metric column
                    col = data[metric]
                    df = pd.concat([df, col], axis=1)
                df.columns = data_name_list
                # Generate boxplot (with legend for each box)
                df.boxplot(column=data_name_list, rot=90)
                # Specify plot labels
                plt.ylabel(str(metric))
                plt.xlabel('Dataset')
                # Export and/or show plot
                plt.savefig(self.experiment_path + '/DatasetComparisons/dataCompBoxplots/DataCompare_' + self.abbrev[
                    algorithm] + '_' + metric + '.png', bbox_inches="tight")
                if self.show_plots:
                    plt.show()
                else:
                    plt.close('all')

    def save_runtime(self):
        """
        Save phase runtime
        """
        runtime_file = open(self.experiment_path + '/runtime/runtime_compare.txt', 'w')
        runtime_file.write(str(time.time() - self.job_start_time))
        runtime_file.close()

    def inter_set_fn(self, fn, algorithm):
        master_list = list()
        for metric in self.metrics:
            for x in range(0, len(self.dataset_directory_paths) - 1):
                for y in range(x + 1, len(self.dataset_directory_paths)):
                    # Grab info on first dataset
                    file1 = self.dataset_directory_paths[x] + '/model_evaluation/' + self.abbrev[
                        algorithm] + '_performance.csv'
                    td1 = pd.read_csv(file1)
                    set1 = td1[metric]
                    med1 = td1[metric].median()
                    # Grab info on second dataset
                    file2 = self.dataset_directory_paths[y] + '/model_evaluation/' + self.abbrev[
                        algorithm] + '_performance.csv'
                    td2 = pd.read_csv(file2)
                    set2 = td2[metric]
                    med2 = td2[metric].median()

                    temp_list = self.temp_summary(set1, set2, x, y, metric, fn)

                    temp_list.append(str(round(med1, 6)))
                    temp_list.append(str(round(med2, 6)))
                    master_list.append(temp_list)
        return master_list

    def inter_set_best_fn(self, fn, global_data):
        label = ['Metric', 'Data1', 'Data2', 'Statistic', 'P-Value', 'Sig(*)']
        for i in range(1, 3):
            label.append('Best_Alg_Data' + str(i))
            label.append('Median_Data' + str(i))

        master_list = list()
        for j in range(len(self.metrics)):
            metric = self.metrics[j]
            for x in range(0, len(self.datasets) - 1):
                for y in range(x + 1, len(self.datasets)):
                    set1 = global_data[j][0][x]
                    med1 = global_data[j][1][x][1]
                    set2 = global_data[j][0][y]
                    med2 = global_data[j][1][y][1]

                    temp_list = self.temp_summary(set1, set2, x, y, metric, fn)

                    temp_list.append(global_data[j][1][x][0])
                    temp_list.append(str(round(med1, 6)))
                    temp_list.append(global_data[j][1][y][0])
                    temp_list.append(str(round(med2, 6)))
                    master_list.append(temp_list)

        # Export analysis summary to .csv file
        df = pd.DataFrame(master_list)
        df.columns = label
        return df

    def temp_summary(self, set1, set2, x, y, metric, fn):

        temp_list = list()
        # handle error when metric values are equal for both algorithms
        if set1.equals(set2):  # Check if all nums are equal in sets
            result = ['NA', 1]
        else:
            try:
                result = fn(set1, set2)
            except Exception:
                result = ['NA_error', 1]
        # Summarize test information in list
        temp_list.append(str(metric))
        temp_list.append('D' + str(x + 1))
        temp_list.append('D' + str(y + 1))
        if set1.equals(set2):
            temp_list.append(result[0])
        else:
            temp_list.append(str(round(result[0], 6)))
        temp_list.append(str(round(result[1], 6)))
        if result[1] < self.sig_cutoff:
            temp_list.append(str('*'))
        else:
            temp_list.append(str(''))

        return temp_list
