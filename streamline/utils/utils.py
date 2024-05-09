import os

def len_datasets(self):
    output_path, experiment_name = self.params['output_path'], self.params['experiment_name']
    datasets = os.listdir(output_path + '/' + experiment_name)
    remove_list = ['.DS_Store', 'metadata.pickle', 'metadata.csv', 'algInfo.pickle',
                'jobsCompleted', 'logs', 'jobs', 'DatasetComparisons',
                'UsefulNotebooks', 'dask_logs',
                experiment_name + '_STREAMLINE_Report.pdf']
    for text in remove_list:
        if text in datasets:
            datasets.remove(text)
    return len(datasets)