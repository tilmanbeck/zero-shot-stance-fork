import os 
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

prediction_path="/storage/ukp/work/beck/Repositories/zero-shot-stance-fork/predictions/"
datasets = ["emergent","argmin","vast","poldeb","mtsd","rumor","wtwt","iac1","scd","semeval2016t6","perspectrum","ibmcs","arc","fnc1","snopes"]
seeds = [0]#[0,1,2,3,4,5,6,7,8,9]
results = []
seed_avgs = []
for seed in seeds:
	seed_results = {'seed': seed}
	seed_avg = []
	for dataset in datasets:
		try:
			p = os.path.join(prediction_path, dataset, str(seed)+"-predictions-test.csv")
			df = pd.read_csv(p)
		except:
			continue
		f1_macro = f1_score(df["label"].values, df["pred label"].values, average="macro")
		seed_results[dataset] = f1_macro
		seed_avg.append(f1_macro)
	results.append(seed_results)
	avg = np.mean(seed_avg)
	seed_avgs.append(avg)
	print('Seed F1-Macro Avg: {:.2f} Std: {:.2f}'.format(avg * 100, np.std(seed_avg) * 100))
print()
print('Overall F1-Macro Avg: {:.2f} Std: {:.2f}'.format(np.mean(seed_avgs)*100, np.std(seed_avgs)*100))
pd.DataFrame(results).to_csv("/storage/ukp/work/beck/Repositories/zero-shot-stance-fork/results.csv")
	



