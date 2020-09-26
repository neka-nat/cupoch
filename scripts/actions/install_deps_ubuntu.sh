#!/usr/bin/env bash
set -ev

sudo apt-get update

sudo apt-get --yes install xorg-dev libglu1-mesa-dev libgl1-mesa-glx || true
sudo apt-get --yes install libglu1-mesa-dev || true
sudo apt-get --yes install libxi-dev || true
sudo apt-get --yes install rename || true
