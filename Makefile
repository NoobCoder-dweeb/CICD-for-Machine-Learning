DATASET = "drug200.csv"

install:
	pip install --upgrade pip && \
		pip install -r requirements.txt

format:
	black *.py

train: 
	python train.py ${DATASET}

eval:
	echo "## Model Metrics" > report.md
	cat ./Results/metrics.txt >> report.md
	echo "" >> report.md
	echo '## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](./Results/confusion_matrix.png)' >> report.md

	cml comment create report.md
