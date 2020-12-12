
clean:
	rm -r -f pyvenv

pyvenv:
	python3 -m venv pyvenv
	pyvenv/bin/pip install --upgrade pip
	pyvenv/bin/pip install -r requirements.txt

check: pyvenv
	pyvenv/bin/flake8 src
	pyvenv/bin/mypy --ignore-missing-imports src

linear: pyvenv
	pyvenv/bin/python src/linear_regression.py
