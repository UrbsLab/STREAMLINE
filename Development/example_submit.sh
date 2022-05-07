#!/usr/bash

set -u

# Define bash variables

data_path=$1
out_path=$2
exp_name=$3
inst_label=$4
class_label=$5


echo "Phase 1: Exploratory Analysis"
python ExploratoryAnalysisMain.py --data-path $data_path --out-path $out_path --exp-name $exp_name --inst-label $inst_label --class-label $class_label --run-parallel False

echo "Phase 2: Data Preprocessing"
python DataPreprocessingMain.py --out-path $out_path --exp-name $exp_name --run-parallel False

echo "Phase 3: Feature Importance Evaluation"
python FeatureImportanceMain.py --out-path $out_path --exp-name $exp_name --run-parallel False

echo "Phase 4: Feature Selection"
python FeatureSelectionMain.py --out-path $out_path --exp-name $exp_name --run-parallel False

echo "Phase 5: Machine Learning Modeling"
python ModelMain.py --out-path $out_path --exp-name $exp_name --run-parallel False

echo "Phase 6: Statistics Summary"
python StatsMain.py --out-path $out_path --exp-name $exp_name --run-parallel False

echo "Phase 7: [Optional] Compare Datasets"
python DataCompareMain.py --out-path $out_path --exp-name $exp_name --run-parallel False

echo "Phase 8: [Optional] Generate PDF Training Summary Report"
python PDF_ReportTrainMain.py --out-path $out_path --exp-name $exp_name --run-parallel False

# echo "Phase 10: [Optional] Apply Models to Replication Data"
#python ApplyModelMain.py --out-path $out_path --exp-name $exp_name --rep-data-path /myrepdatapath/TestRep  --data-path ${data_path}/hcc-data_example.csv --run-parallel False

# echo "Phase 11: [Optional] Generate PDF 'Apply Replication' Summary Report"
#python PDF_ReportApplyMain.py --out-path $out_path --exp-name $exp_name --rep-data-path /myrepdatapath/TestRep  --data-path ${data_path}/hcc-data_example.csv --run-parallel False
