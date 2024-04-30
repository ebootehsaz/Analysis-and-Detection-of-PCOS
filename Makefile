.PHONY: test

test:
	autopep8 --in-place --recursive --max-line-length=120 src/
	flake8 src/ --max-line-length=120
	
