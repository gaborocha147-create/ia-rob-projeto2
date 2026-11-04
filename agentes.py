"""Q-learning e Sarsa usados nos notebooks do projeto.

Os dois agentes funcionam do mesmo jeito: têm uma Q-table começando em zero,
escolhem ações com ε-greedy e diminuem o ε a cada episódio. A única diferença
é a fórmula que atualiza a Q-table.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from numpy import savetxt


def criar_env(**env_kwargs):
    # .env tira o limite de 200 passos: no treino o episódio só acaba na entrega
    return gym.make("Taxi-v4", render_mode='ansi', **env_kwargs).env


class Agente:
    """Parte comum do Q-learning e do Sarsa."""

    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes, seed=None):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        # com a mesma seed, o treino sorteia sempre os mesmos números
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.rewards_per_episode   = []
        self.timesteps_per_episode = []

    def select_action(self, state):
        # ε-greedy: com chance ε explora (ação aleatória), senão usa a melhor ação da Q-table
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.action_space.n))
        return int(np.argmax(self.q_table[state]))

    def train(self, filename=None, plotFile=None, rewardFile=None, timestepFile=None, log_every=2500):
        for i in range(1, self.episodes + 1):
            # só o primeiro episódio recebe a seed; os outros continuam a mesma sequência
            (state, _) = self.env.reset(seed=self.seed if i == 1 else None)

            rewards, timesteps = self.run_episode(state)
            self.timesteps_per_episode.append(timesteps)
            self.rewards_per_episode.append(rewards)

            if log_every and i % log_every == 0:
                avg = np.mean(self.rewards_per_episode[-100:])
                print(f"Episódio: {i}/{self.episodes} | ε={self.epsilon:.4f} | "
                      f"reward médio (últimos 100): {avg:.2f}")

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_dec

        if filename:
            savetxt(filename, self.q_table, delimiter=',')
        if plotFile is not None:
            self.plotactions(plotFile, self.timesteps_per_episode)
        if rewardFile is not None:
            self.plotrewards(rewardFile)
        if timestepFile is not None:
            self.plottimesteps(timestepFile)

        return self.q_table

    def plotactions(self, plotFile, actions_per_episode):
        plt.figure()
        plt.plot(actions_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Actions')
        plt.title('# Actions vs Episodes')
        plt.savefig(plotFile + ".jpg")
        plt.close()

    def plotrewards(self, rewardFile):
        plt.figure()
        plt.plot(self.rewards_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.title('Reward vs Episodes')
        plt.savefig(rewardFile + ".jpg")
        plt.close()

    def plottimesteps(self, timestepFile):
        plt.figure()
        plt.plot(self.timesteps_per_episode)
        plt.xlabel('Episodes')
        plt.ylabel('# Timesteps')
        plt.title('# Timesteps vs Episodes')
        plt.savefig(timestepFile + ".jpg")
        plt.close()


class QLearning(Agente):
    def run_episode(self, state):
        rewards, timesteps, done = 0, 0, False
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)

            # Q(s,a) = Q(s,a) + α * (r + γ * max Q(s',a') - Q(s,a))
            # usa a MELHOR ação do próximo estado, mesmo que o agente não vá fazê-la
            next_max = np.max(self.q_table[next_state]) * (not done)  # depois da entrega não há futuro
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

            state = next_state
            timesteps += 1
            rewards += reward
        return rewards, timesteps


class Sarsa(Agente):
    def run_episode(self, state):
        rewards, timesteps, done = 0, 0, False
        action = self.select_action(state)
        while not done:
            next_state, reward, done, _, _ = self.env.step(action)

            # Q(s,a) = Q(s,a) + α * (r + γ * Q(s',a') - Q(s,a))
            # usa a ação a' que o agente REALMENTE vai fazer (pode ser aleatória)
            next_action = self.select_action(next_state)
            next_value = self.q_table[next_state, next_action] * (not done)  # depois da entrega não há futuro
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.alpha * (reward + self.gamma * next_value - old_value)

            state, action = next_state, next_action
            timesteps += 1
            rewards += reward
        return rewards, timesteps


def avaliar_q_table(q_table, n_eval=100, max_steps=200, **env_kwargs):
    """Testa uma Q-table sem ações aleatórias, sempre escolhendo a melhor ação.

    O episódio número ep usa seed=ep, então todas as Q-tables são testadas nos
    mesmos episódios. Episódios que passam de max_steps contam em 'Não terminou'.
    """
    env = criar_env(**env_kwargs)
    passos, rewards, pens, nao_terminou = [], [], [], 0

    for ep in range(n_eval):
        state, _ = env.reset(seed=ep)
        done, steps, ep_reward, pen = False, 0, 0, 0
        while not done and steps < max_steps:
            state, reward, done, _, _ = env.step(np.argmax(q_table[state]))
            steps += 1
            ep_reward += reward
            pen += (reward == -10)
        nao_terminou += (not done)
        passos.append(steps); rewards.append(ep_reward); pens.append(pen)

    env.close()
    return {'Timesteps (média)': np.mean(passos),
            'Desvio padrão': np.std(passos),
            'Mín': float(np.min(passos)), 'Máx': float(np.max(passos)),
            'Reward médio': np.mean(rewards),
            'Penalidades': np.mean(pens),
            'Não terminou': float(nao_terminou)}


def avaliar(pasta, prefixo, n_eval=100, max_steps=200, n_runs=5, **env_kwargs):
    """Testa as 5 Q-tables salvas em pasta e monta a tabela com a média no fim."""
    linhas = []
    for run in range(1, n_runs + 1):
        q_table = np.loadtxt(f'{pasta}/q-table-taxi-driver-{prefixo}-run{run}.csv', delimiter=',')
        linhas.append({'Execução': str(run), **avaliar_q_table(q_table, n_eval, max_steps, **env_kwargs)})

    df = pd.DataFrame(linhas)
    media = df.drop(columns='Execução').mean()
    media['Mín'], media['Máx'] = df['Mín'].min(), df['Máx'].max()
    df.loc['Média'] = {'Execução': 'Média', **media.to_dict()}
    return df.round(2)
