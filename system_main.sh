#!/bin/bash
source ~/.bashrc
cd /home/skm21/OEIS_Sequence_Formula_discovery-main
source /home/skm21/miniconda3/bin/activate OEIS
python system_main.py   >> /home/skm21/OEIS_Sequence_Formula_discovery-main/service.log 2>&1