# MountainCar-ModularRL

## Requirements
python3

1. numpy==1.17.2
2. torch==1.3.0
3. matplotlib==3.0.3

## Instructions
Implemented three RL algorithms
1. Q-Learning using Q-Table
run 
python
```
python3 Q_learning.py
```

2. Deep Q-Learning without Experience Replay
run
python
```
python3 deep_QN_noExpRepl.py
```

3. Deep Q-learning with Experience Replay and reward shaping
run (this runs both fuel experiments with and without fuel consideration)
python
```
python3 deep_QN_ExpReplay.py
```

All three experiments produce figures showing reward progress over episodes and final car position over episodes.
Experiment 3 also shows commulative action distribution.
