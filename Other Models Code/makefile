install: venv
	./venv/bin/activate; pip3 install -Ur requirements.txt
venv :
	test -d venv || python3 -m venv venv
clean:
	rm -rf venv
run:
	python3 src/main.py
test:
	python3 src/test.py
