g++ -o main -I/home/pi/tensorflow_src -I/home/pi/tflite_build/flatbuffers/include -ltensorflowlite `pkg-config --cflags --libs opencv` main.cpp


