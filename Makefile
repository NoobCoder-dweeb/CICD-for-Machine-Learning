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

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:update