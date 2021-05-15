#! /bin/bash

echo "=============aue============="
python run.py --model_name="aue"
echo "=============awe============="
python run.py --model_name="awe"
echo "=============sea============="
python run.py --model_name="sea"
echo "=============online bagging============="
python run.py --model_name="onlinebagging"
echo "=============mlp============="
python run.py --model_name="mlp"