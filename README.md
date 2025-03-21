# Pokémon Red - Reinforcement Learning  

This project applies **Reinforcement Learning (RL)** to the classic **Pokémon Red** game using emulation and AI agents. The goal is to train an AI to play the game effectively by learning through trial and error.  

## 📌 Features  
- Uses **Reinforcement Learning** techniques (Q-Learning, DQN, or PPO).  
- Direct interaction with the **Pokémon Red** ROM through an emulator.  
- State representation based on in-game variables (HP, position, battle state, etc.).  
- Training AI to make optimal decisions in battles, movement, and objectives.  
- Python-based **environment wrapper** for controlled interactions.  

## 🛠️ Tech Stack  
- **Python** (Main language)  
- **OpenAI Gymnasium** (For environment simulation)  
- **Stable-Baselines3** (For RL algorithms)  
- **PyTorch / TensorFlow** (For deep learning models)  

## How It Works  
1. The emulator runs **Pokémon Red** with an RL agent controlling the game.  
2. The AI **observes game states** (player position, Pokémon HP, enemy stats, deaths, new exploration, badges ).  
3. It **selects actions** (move, attack, switch Pokémon) based on a reward system.  
4. The AI **learns** over time to optimize gameplay decisions.
5. Uses Custom Gymnasium API to calculate and process **Rewards** foundation.

Just remember, if you're trying to run this in the future then you'll see some of the packages or modules will be out of date so make sure to update them.
**Peace!**
