import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt
t = time.process_time()

class Genetic:
    def __init__(self, env, number_actions, obs_space_size, generations, mutation_prop=0.05, top_hypotheses=100):
        self.env = env
        self.num_act = number_actions
        self.frame_size = obs_space_size
        self.model = self.make_model()
        self.generations = generations
        self.scores = []
        self.weight_map = dict()
        self.mutation_prop = mutation_prop
        self.generation_reward_track = []
        self.final_weights = ()
        self.top_hypotheses = top_hypotheses
        self.last_max=0

    def make_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(10,activation='relu',input_shape=[2,1]))
        model.add(tf.keras.layers.Dense(units=3,activation='softmax'))
        return model

    def evaluate_hypothesis(self, weights):
        i = 0
        for layer in self.model.layers:
            current_weights = layer.get_weights()
            if not current_weights:
                continue
            if len(current_weights) == 2:
                layer.set_weights([weights[i], current_weights[1]])
                i += 1
            elif len(current_weights) == 1:
                layer.set_weights([weights[i]])
                i += 1
        score, time_steps = 0, 0
        terminal = False
        S, _ = self.env.reset()
        while not terminal:
            S=tf.expand_dims(S,axis=-1)
            action_probs = self.model.predict(S)
            print(action_probs[0][0])
            action = np.random.choice(3,p=action_probs[0][0])
            S_, R, T, _, _ = self.env.step(action)
            S=S_
            print(f"action:{action},reward:{R}")
            terminal = T
            score += R
            time_steps += 1
            if time_steps > 30 or terminal:
                break
        self.weight_map[score] = weights
        return score

    def initialize_random_hypothesis(self):
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(self.model.summary())
        weights = [layer.get_weights()[0] for layer in self.model.layers if layer.get_weights()]
        print(len(weights))
        num_hypos = 4
        W1, W2,= [], []
        for _ in range(num_hypos):
            W1.append(np.random.uniform(-2, 2, size=weights[0].shape))
            W2.append(np.random.uniform(-2, 2, size=weights[1].shape))
        W1 = np.array(W1)
        W2 = np.array(W2)
        for w1 in W1:
            for w2 in W2:
                w_tup = (w1, w2)
                score = self.evaluate_hypothesis(w_tup)
                self.scores.append(score)

        sorted_scores = sorted(self.scores, reverse=True)
        top_scores = sorted_scores[:self.top_hypotheses]
        top_weights = [self.weight_map[score] for score in top_scores]

        self.scores = top_scores
        self.weight_map = {score: weight for score, weight in zip(top_scores, top_weights)}

        self.populate()

    def populate(self):
        for i in range(self.generations):
            maxi = max(self.scores)
            self.final_weights=self.weight_map[maxi]
            self.last_max=max(self.last_max,maxi)
            self.generation_reward_track.append(maxi)
            print(f"Maximum reward for generation {i} is {maxi}")
            print(f"Mutation prop {self.mutation_prop}")
            with open("scores_by_ep.txt", "a") as file:
                file.write(f"{maxi}\n")
            chosen_parents = self.select_parents()
            self.mutate_parents(chosen_parents)

    def select_parents(self):
        mini = min(self.scores)
        d_scores = [i+abs(mini)+1 for i in self.scores]
        num_mutated_hypos = int(self.mutation_prop * len(self.scores))
        chosen_indices = []
        for _ in range(min(num_mutated_hypos, len(d_scores))):
            selection_prob = [cur_score / sum(d_scores) for cur_score in d_scores]
            ind = np.random.choice(len(d_scores), p=selection_prob)
            chosen_indices.append(ind)
            del d_scores[ind]
        chosen_scores = [self.scores[ind] for ind in chosen_indices]

        return chosen_scores

    def mutate_parents(self, chosen_scores):
        cur_max=0
        for score in chosen_scores:
            if score not in self.weight_map: continue
            W1, W2 = self.weight_map[score]
            W1_M1=np.random.permutation(W1)
            W1_M2=W1*np.random.choice([-1, 1], size=W1.shape)
            W2_M1=np.random.permutation(W2)
            W2_M2=W2*np.random.choice([-1, 1], size=W2.shape)
            score1 = self.evaluate_hypothesis((W1_M1, W2_M1))
            self.scores.append(score1)
            score2 = self.evaluate_hypothesis((W1_M2, W2_M2))
            self.scores.append(score2)
            cur_max=max(cur_max,score1,score2)
            sorted_scores = sorted(self.scores, reverse=True)
            top_scores = sorted_scores[:self.top_hypotheses]
            top_weights = [self.weight_map[score] for score in top_scores]
            self.scores = top_scores
            self.weight_map = {score: weight for score, weight in zip(top_scores, top_weights)}
        if cur_max<=self.last_max:
            self.mutation_prop=min(1,self.mutation_prop*1.1)
        else : self.mutation_prop*=0.9
