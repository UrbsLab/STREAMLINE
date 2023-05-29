python run.py --data-path ./data/DemoData --out-path demo --exp-name demo_full --do-till-report --class-label Class --inst-label InstanceID --algorithms=NB,LR,DT --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-eda --class-label Class --inst-label InstanceID --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-dataprep --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-feat-imp --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-feat-sel --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-model --algorithms NB,LR,DT --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-stats --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-compare-dataset --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-report --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-replicate --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-rep-report --run-cluster SLURMOld --res-mem 4 --queue defq
python run.py --out-path demo --exp-name demo --do-clean --del-time --del-old-cv --run-cluster SLURMOld --r