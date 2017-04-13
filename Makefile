all: install

install:
	python setup.py install

test:
	pytest nfft

test-cov:
	py.test --cov-report term-missing --cov=nfft nfft
