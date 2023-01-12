from streamline.utils.job import Job
from streamline.utils.dataset import Dataset


class KFoldPartitioner(Job):
    """
    Base class for KFold CrossValidation Operations on dataset
    """
    def __init__(self, dataset, partition_method, experiment_path):
        """
        Initialization for KFoldPartitioner base class

        Args:
            dataset: a streamline.utils.dataset.Dataset object or a path to dataset text file
            experiment_path: path to experiment the logging directory folder
        """
        super().__init__()
        if type(dataset) == str:
            self.dataset = Dataset(dataset)
        else:
            assert (type(dataset) == Dataset)
            self.dataset = dataset
        self.dataset_path = dataset.path
        self.experiment_path = experiment_path
        self.partition_method = partition_method

# def cv_partitioner(data, cv_partitions, partition_method, class_label, match_label, randomSeed):
#     """ Takes data frame (data), number of cv partitions, partition method (R, S, or M), class label,
#     and the column name used for matched CV. Returns list of training and testing dataframe partitions."""
#     # Shuffle instances to avoid potential order biases in creating partitions
#     data = data.sample(frac=1, random_state=randomSeed).reset_index(drop=True)
#     # Convert data frame to list of lists (save header for later)
#     header = list(data.columns.values)
#     datasetList = list(list(x) for x in zip(*(data[x].values.tolist() for x in data.columns)))
#     outcomeIndex = data.columns.get_loc(class_label)  # Get classIndex
#     if not match_label == 'None':
#         matchIndex = data.columns.get_loc(match_label)  # Get match variable column index
#     classList = pd.unique(data[class_label]).tolist()
#     del data  # memory cleanup
#     # Initialize partitions-----------------------------
#     partList = []  # Will store partitions
#     for x in range(cv_partitions):
#         partList.append([])
#     # Random Partitioning Method----------------------------------------------------------------
#     if partition_method == 'R':
#         currPart = 0
#         counter = 0
#         for row in datasetList:
#             partList[currPart].append(row)
#             counter += 1
#             currPart = counter % cv_partitions
#     # Stratified Partitioning Method------------------------------------------------------------
#     elif partition_method == 'S':
#         # Create data sublists, each having all rows with the same class
#         byClassRows = [[] for i in range(len(classList))]  # create list of empty lists (one for each class)
#         for row in datasetList:
#             # find index in classList corresponding to the class of the current row.
#             cIndex = classList.index(row[outcomeIndex])
#             byClassRows[cIndex].append(row)
#         for classSet in byClassRows:
#             currPart = 0
#             counter = 0
#             for row in classSet:
#                 partList[currPart].append(row)
#                 counter += 1
#                 currPart = counter % cv_partitions
#     # Matched partitioning method ---------------------------------------------------------------
#     elif partition_method == 'M':
#         # Create data sublists, each having all rows with the same match identifier
#         matchList = []
#         for each in datasetList:
#             if each[matchIndex] not in matchList:
#                 matchList.append(each[matchIndex])
#         byMatchRows = [[] for i in range(len(matchList))]  # create list of empty lists (one for each match group)
#         for row in datasetList:
#             # find index in matchList corresponding to the matchset of the current row.
#             mIndex = matchList.index(row[matchIndex])
#             row.pop(matchIndex)  # remove match column from partition output
#             byMatchRows[mIndex].append(row)
#         currPart = 0
#         counter = 0
#         for matchSet in byMatchRows:  # Go through each unique set of matched instances
#             for row in matchSet:  # put all of the instances
#                 partList[currPart].append(row)
#             # move on to next matchset being placed in the next partition.
#             counter += 1
#             currPart = counter % cv_partitions
#         header.pop(matchIndex)  # remove match column from partition output
#     else:
#         raise Exception('Error: Requested partition method not found.')
#     del datasetList  # memory cleanup
#     # Create (cv_partitions) training and testing sets from partitions -------------------------------------------
#     train_dfs = []
#     test_dfs = []
#     for part in range(0, cv_partitions):
#         testList = partList[part]  # Assign testing set as the current partition
#         trainList = []
#         tempList = []
#         for x in range(0, cv_partitions):
#             tempList.append(x)
#         tempList.pop(part)
#         for v in tempList:  # for each training partition
#             trainList.extend(partList[v])
#         train_dfs.append(pd.DataFrame(trainList, columns=header))
#         test_dfs.append(pd.DataFrame(testList, columns=header))
#     del partList  # memory cleanup
#     return train_dfs, test_dfs
#
#
# def saveCVDatasets(experiment_path, dataset_name, train_dfs, test_dfs):
#     """ Saves individual training and testing CV datasets as .csv files"""
#     # Generate folder to contain generated CV datasets
#     if not os.path.exists(experiment_path + '/' + dataset_name + '/CVDatasets'):
#         os.mkdir(experiment_path + '/' + dataset_name + '/CVDatasets')
#     # Export training datasets
#     counter = 0
#     for each in train_dfs:
#         a = each.values
#         with open(experiment_path + '/' + dataset_name + '/CVDatasets/' + dataset_name + '_CV_' + str(
#                 counter) + "_Train.csv", mode="w", newline="") as file:
#             writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             writer.writerow(each.columns.values.tolist())
#             for row in a:
#                 writer.writerow(row)
#         counter += 1
#     # Export testing datasets
#     counter = 0
#     for each in test_dfs:
#         a = each.values
#         with open(experiment_path + '/' + dataset_name + '/CVDatasets/' + dataset_name + '_CV_' + str(
#                 counter) + "_Test.csv", mode="w", newline="") as file:
#             writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#             writer.writerow(each.columns.values.tolist())
#             for row in a:
#                 writer.writerow(row)
#         file.close()
#         counter += 1