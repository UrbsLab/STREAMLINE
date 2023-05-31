rm -rf ./demo/demo
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-eda --class-label Class --inst-label InstanceID --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-dataprep --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-feat-imp --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-feat-sel --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-model --algorithms NB,LR,DT --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-stats --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-compare-dataset --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-report --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-replicate --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-rep-report --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-clean --del-time --del-old-cv --run-cluster LSFOld
rm -rf ./demo/demo
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-eda --class-label Class --inst-label InstanceID --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-dataprep --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-feat-imp --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-feat-sel --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-model --algorithms NB,LR,DT --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-stats --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-compare-dataset --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-report --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-replicate --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --rep-path ./data/DemoRepData --dataset ./data/DemoData/hcc-data_example_custom.csv --out-path demo --exp-name demo --do-rep-report --run-cluster LSF --res-mem 4 --queue i2c2_normal
python run.py --out-path demo --exp-name demo --do-clean --del-time --del-old-cv --run-cluster LSF
rm -rf ./demo/demo
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-till-report --class-label Class --inst-label InstanceID --algorithms=NB,LR,DT --run-cluster LSFOld --res-mem 4 --queue i2c2_normal
rm -rf ./demo/demo
python run.py --data-path ./data/DemoData --out-path demo --exp-name demo --do-till-report --class-label Class --inst-label InstanceID --algorithms=NB,LR,DT --run-cluster LSF --res-mem 4 --queue i2c2_normal
rm -rf ./demo/demo
python run.py -c upenn.cfg