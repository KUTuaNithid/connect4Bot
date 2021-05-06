# connect4Bot

This project is created to study about deploying the AI system in to embedded system. We are interetested in a reinforcement, so we decide to create a Connect4 AI player follow AlphaZero.
https://medium.com/aiothardwaredesign/connect4bot-384a1a124e03

## How to use

### Training
```
cd src/reinforcement
python3 train_player.py
```
The created model will be created in reinforcement/Models folder

### Play with AI
```
cd src/reinforcement
python3 playVsAI.py
```
You need to change the paremeter in `ZeroAI =ZeroBrain("model_name")` to your model.

### Run on coral
```
cd src
python3 Main_play.py
```
