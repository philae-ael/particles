debug: main.cpp
	g++ -lSDL3 -g -march=native -o debug main.cpp
run-debug: debug
	./debug

release: main.cpp
	g++ -lSDL3 -O3 -march=native -o release main.cpp
run-release: release
	./release
