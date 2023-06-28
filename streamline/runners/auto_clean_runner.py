class AutoCleanRunner: 

    def __init__(self):

        #Dataprocess_runner 
        self.data_path = data_path
        self.output_path = output_path
        self.experiment_name = experiment_name
        self.class_label = class_label
        self.instance_label = instance_label
        self.match_label = match_label
        self.ignore_features = ignore_features
        self.categorical_cutoff = 'Optuna'
        self.categorical_features = categorical_features
        self.quantitative_features = quantitative_features
        self.featureeng_missingness = 'Optuna'
        self.cleaning_missingness = 'Optuna'
        self.correlation_removal_threshold = 'Optuna'
        self.top_features = top_features
        self.exploration_list = 'Optuna'
        self.plot_list = plot_list
        self.n_splits = 'Optuna'
        self.partition_method = 'Optuna'
        self.run_cluster = run_cluster
        self.queue = queue
        self.reserved_memory = reserved_memory
        self.show_plots = show_plots

        #ImputationRunner
        self.scale_data = scale_data
        self.impute_data = impute_data
        self.multi_impute = multi_impute
        self.overwrite_cv = overwrite_cv
        self.random_state = random_state
        
        #FeatureImportanceRunner
        self.cv_count = None
        self.dataset = None
        self.instance_subset = instance_subset
        self.algorithms = list(algorithms)
        self.use_turf = use_turf
        self.turf_pct = turf_pct
        self.n_jobs = n_jobs

        #FeatureSelectionRunner
        self.cv_count = None
        self.dataset = None
        self.max_features_to_keep = max_features_to_keep
        self.filter_poor_features = filter_poor_features
        self.export_scores = export_scores
    
    def run(self):
