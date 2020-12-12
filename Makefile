clean:
	rm -r -f {__pycache__,data,pyvenv}

pyvenv:
	python3 -m venv pyvenv
	pyvenv/bin/pip install --upgrade pip
	pyvenv/bin/pip install -r requirements.txt

check: pyvenv
	pyvenv/bin/flake8 src
	pyvenv/bin/mypy --ignore-missing-imports src

data:
	mkdir data
	curl https://www.clips.uantwerpen.be/conll2000/chunking/train.txt.gz --output data/conll2000_chunking_train.txt.gz
	gunzip data/conll2000_chunking_train.txt.gz
	curl https://www.clips.uantwerpen.be/conll2000/chunking/test.txt.gz --output data/conll2000_chunking_test.txt.gz
	gunzip data/conll2000_chunking_test.txt.gz

linear: pyvenv
	pyvenv/bin/python src/linear_regression.py
