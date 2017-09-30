#! /bin/bash

g++ Main.cpp -o image_registration `pkg-config --cflags --libs opencv`