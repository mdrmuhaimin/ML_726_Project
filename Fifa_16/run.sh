#!/usr/bin/env bash
unzip input/Player_Att.zip
mv database.sqlite input/
echo 'Starting program'
spark-submit --driver-class-path sqlite-jdbc-3.21.0.jar fifa16_Processor.py
rm input/database.sqlite