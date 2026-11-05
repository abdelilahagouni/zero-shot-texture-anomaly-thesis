@echo off
rem Resume the feature-layer sweep (finished units are skipped). Then run the analysis stage separately:
rem   C:\Python313\python.exe -u src\feature_layer_sweep.py --stage analyze
cd /d "C:\Users\abdel\OneDrive\Desktop\zero shot\zero_shot_texture_anomaly"
set PYTHONIOENCODING=utf-8
C:\Python313\python.exe -u src\feature_layer_sweep.py --stage full
pause
