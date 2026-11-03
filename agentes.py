"""Agentes tabulares (Q-learning, Sarsa e Expected Sarsa) usados nos notebooks.

Todos compartilham a mesma estrutura: Q-table iniciada com zeros, política
ε-greedy com decaimento do ε a cada episódio e registro do reward e do número
de passos de cada episódio. A diferença entre eles está só no alvo da
atualização da Q-table.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
import matplotlib.pyplot as plt
from numpy import savetxt


def criar_env(**env_kwargs):
    # .env remove o TimeLimit: no treino o episódio só termina na entrega
    return gym.make("Taxi-v4", render_mode='ansi', **env_kwargs).env


class AgenteTabular:
    def __init__(self, env, alpha, gamma, epsilon, epsilon_min, epsilon_dec, episodes, seed=None):
        self.env = env
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.episodes = episodes
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.rewards_per_episode   = []
        self.timesteps_per_episode = []
        self.epsilon_per_episode   = []
        # quantas vezes cada ação foi executada em cada estado durante o treino
        self.visits = np.zeros([env.observation_space.n, env.action_space.n], dtype=np.int64)

    def select_action(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.env.action_space.n))
        return int(np.argmax(self.q_table[state]))

    def run_episode(self, state):
        """Roda um episódio atualizando a Q-table e devolve (reward, timesteps)."""
        raise NotImplementedError

    def train(self, filename=None, plotFile=None, rewardFile=None, timestepFile=None, log_every=2500, callback=None):
        """Treina por self.episodes episódios.

        callback(episodio, agente), se passado, é chamado no fim de cada
        episódio (antes do decaimento do ε) e não interfere no treino.
        """
        for i in range(1, self.episodes + 1):
            # só o primeiro reset recebe a seed; os seguintes continuam a mesma
            # sequência aleatória, então o treino inteiro é reproduzível
            (state, _) = self.env.reset(seed=self.seed if i == 1 else None)
            self.epsilon_per_episode.append(self.epsilon)

            rewards, timesteps = self.run_episode(state)
            self.timesteps_per_episode.append(timesteps)
            self.rewards_per_episode.append(rewards)

            if log_every and i % log_every == 0:
                avg = np.mean(self.rewards_per_episode[-100:])
                print(f"Episódio: {i}/{self.episodes} | ε={self.epsilon:.4f} | "
                      f"reward médio (últimos 100): {avg:.2f}")

            if callback is not None:
                callback(i, self)

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


class QLearning(AgenteTabular):
    """Off-policy: o alvo usa a melhor ação do próximo estado."""

    def run_episode(self, state):
        rewards, timesteps, done = 0, 0, False
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.visits[state, action] += 1

            next_max = np.max(self.q_table[next_state]) * (not done)
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

            state = next_state
            timesteps += 1
            rewards += reward
        return rewards, timesteps


class Sarsa(AgenteTabular):
    """On-policy: o alvo usa a ação que o agente realmente vai executar."""

    def run_episode(self, state):
        rewards, timesteps, done = 0, 0, False
        action = self.select_action(state)
        while not done:
            next_state, reward, done, _, _ = self.env.step(action)
            self.visits[state, action] += 1

            next_action = self.select_action(next_state)
            next_value = self.q_table[next_state, next_action] * (not done)
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.alpha * (reward + self.gamma * next_value - old_value)

            state, action = next_state, next_action
            timesteps += 1
            rewards += reward
        return rewards, timesteps


class ExpectedSarsa(AgenteTabular):
    """Alvo é o valor esperado do próximo estado sob a política ε-greedy atual."""

    def run_episode(self, state):
        rewards, timesteps, done = 0, 0, False
        n_actions = self.env.action_space.n
        while not done:
            action = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            self.visits[state, action] += 1

            q_next = self.q_table[next_state]
            probs = np.full(n_actions, self.epsilon / n_actions)
            probs[np.argmax(q_next)] += 1 - self.epsilon
            expected = probs @ q_next * (not done)
            old_value = self.q_table[state, action]
            self.q_table[state, action] = old_value + self.alpha * (reward + self.gamma * expected - old_value)

            state = next_state
            timesteps += 1
            rewards += reward
        return rewards, timesteps


def avaliar_q_table(q_table, n_eval=100, max_steps=200, **env_kwargs):
    """Testa uma Q-table usando só a política gulosa (sem exploração).

    O episódio ep sempre usa seed=ep, então Q-tables diferentes são testadas
    nos mesmos episódios. Episódios que passam de max_steps são contados em
    'Não terminou'.
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
    """Avalia as Q-tables salvas em pasta (uma por execução) com avaliar_q_table."""
    linhas = []
    for run in range(1, n_runs + 1):
        q_table = np.loadtxt(f'{pasta}/q-table-taxi-driver-{prefixo}-run{run}.csv', delimiter=',')
        linhas.append({'Execução': str(run), **avaliar_q_table(q_table, n_eval, max_steps, **env_kwargs)})

    df = pd.DataFrame(linhas)
    media = df.drop(columns='Execução').mean()
    media['Mín'], media['Máx'] = df['Mín'].min(), df['Máx'].max()
    df.loc['Média'] = {'Execução': 'Média', **media.to_dict()}
    return df.round(2)
