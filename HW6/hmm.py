import math

import numpy as np

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    if hasattr(math, 'isclose'):
        return math.isclose(a, b)
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

class HMM:
    def __init__(self, initial_probs, transition_probs, emission_probs, state_labels, emission_labels):
        # sanity check
        # no fitting, fixed markov model
        assert transition_probs is not None
        assert emission_probs is not None

        assert len(emission_probs) == len(transition_probs)
        assert isclose(sum(initial_probs), 1.0)
        for state_transfer_probs in transition_probs:
            assert len(state_transfer_probs) == len(transition_probs)
            assert isclose(sum(state_transfer_probs), 1.0)
        for state_emission_probs in emission_probs:
            assert len(state_emission_probs) == len(emission_probs[0])
            assert isclose(sum(state_emission_probs), 1.0)
        if state_labels is not None:
            assert len(state_labels) == len(transition_probs)
        if emission_labels is not None:
            assert len(emission_labels) == len(emission_probs[0])
        self.initial_probs = initial_probs
        self.transition_probs = transition_probs
        self.emission_probs = emission_probs
        self.emission_labels = emission_labels
        self.state_labels = state_labels

        self.emission_label_to_idx = {}
        self.state_label_to_idx = {}
        for i, state_label in enumerate(state_labels):
            self.state_label_to_idx[state_label] = i
        for i, em_label in enumerate(emission_labels):
            self.emission_label_to_idx[em_label] = i

    def beta_t_helper(self, t, state_idx, emission_idx_seq, cache):
        assert t >= 0
        if t >= len(emission_idx_seq):
            return 1
        if cache[t][state_idx] >= 0:
            return cache[t][state_idx]

        prob = 0
        if t == len(emission_idx_seq)-1:
            prob = 1
        else:
            prob = 0
            for tp1_state in range(0, len(self.transition_probs)):
                prob += self.transition_probs[state_idx][tp1_state] * self.beta_t_helper(t+1, tp1_state, emission_idx_seq, cache) * self.emission_probs[tp1_state][emission_idx_seq[t+1]]

        cache[t][state_idx] = prob
        return prob

    def alpha_t_helper(self, t, state_idx, emission_idx_seq, cache = None): #cache-> array of t x states
        assert t >= 0
        if cache is None:
            cache = np.full([len(emission_idx_seq), len(self.transition_probs)], -1.0)
        if cache[t][state_idx] >= 0:
            return cache[t][state_idx]

        prob = 0
        if t == 0:
            prob = self.emission_probs[state_idx][emission_idx_seq[t]] * self.initial_probs[state_idx]
        else:
            prob = 0
            for tm1_state in range(0, len(self.transition_probs)):
                prob += self.transition_probs[tm1_state][state_idx] * self.alpha_t_helper(t-1, tm1_state, emission_idx_seq, cache)
            prob *= self.emission_probs[state_idx][emission_idx_seq[t]]
        cache[t][state_idx] = prob
        return prob

    def _get_emission_idx_seq_from_label_seq(self, label_seq):
        emission_idx_list = []
        for op in label_seq:
            emission_idx_list.append(self.emission_label_to_idx[op])
        return emission_idx_list

    def _get_state_label_seq_from_index(self, state_label_idx_list):
        state_label_list = []
        for i in state_label_idx_list:
            state_label_list.append(self.state_labels[i])
        return state_label_list

    def calc_prob_output_sequence(self, output_seq):
        """
        Calculate probability of a given output sequence.
        :param output_seq: A list of output.
        :return: Probability of the given sequence of outputs.
        """
        emission_idx_list = self._get_emission_idx_seq_from_label_seq(output_seq)
        t = len(output_seq)
        cache = np.full([t, len(self.transition_probs)], -1.0)
        prob = 0
        for state in range(0, len(self.transition_probs)):
            prob += self.alpha_t_helper(t-1, state, emission_idx_list, cache)
        return prob

    def get_likelihood(self, t, state_label, emission_seq):
        emission_idx_list = self._get_emission_idx_seq_from_label_seq(emission_seq)
        state_idx = self.state_label_to_idx[state_label]

        cache_a = np.full([len(emission_seq), len(self.transition_probs)], -1.0)
        cache_b = np.full([len(emission_seq), len(self.transition_probs)], -1.0)
        prob = self.alpha_t_helper(t, state_idx, emission_idx_list, cache_a) * self.beta_t_helper(t, state_idx, emission_idx_list, cache_b)
        dnm = 0
        for j in range(0, len(self.transition_probs)):
            dnm += self.alpha_t_helper(t, j, emission_idx_list, cache_a) * self.beta_t_helper(t, j, emission_idx_list, cache_b)
        return prob / dnm

    def get_most_likely_state_seq(self, emission_seq):
        num_states = len(self.transition_probs)
        assert len(emission_seq) > 0
        state_probs = [self.initial_probs]
        parent_state = [[None]*num_states]
        for i in range(0, len(state_probs[0])):
            # TODO: use log domain, underflow possible
            state_probs[0][i] = state_probs[0][i]*self.emission_probs[i][emission_seq[0]]
        for t in range(1, len(emission_seq)):
            parent_state.append([None]*num_states)
            state_probs.append([0]*num_states)
            for tm1state in range(0, num_states):
                for tstate in range(0, num_states):
                    prob = self.transition_probs[tm1state][tstate] * state_probs[t-1][tm1state] * self.emission_probs[tstate][emission_seq[t]]
                    if prob > state_probs[t][tstate]:
                        state_probs[t][tstate] = prob
                        parent_state[t][tstate] = tm1state
        state_seq = []
        t = len(emission_seq) - 1
        while t >= 0:
            cur_state = max(enumerate(state_probs[t]), key=(lambda x: x[1]))[0]
            assert cur_state is not None
            state_seq.append(cur_state)
            t -= 1

        return list(reversed(self._get_state_label_seq_from_index(list(state_seq))))

    def get_most_likely_state_seq_from_labels(self, emission_labels):
        return self.get_most_likely_state_seq(self._get_emission_idx_seq_from_label_seq(emission_labels))

def test_hmm():
    print("\nRunning tests")
    hmm = HMM(initial_probs=[0.8, 0.2]
              , transition_probs= [[0.6, 0.4], [0.3, 0.7]]
              , emission_probs = [[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]]
              , state_labels = ['S1', 'S2']
              , emission_labels = ['R', 'W', 'B'])
    cache = np.full([3, len(hmm.transition_probs)], -1.0)
    assert isclose(hmm.alpha_t_helper(2, 0, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.0162)
    assert isclose(hmm.alpha_t_helper(2, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.01764)
    assert isclose(hmm.alpha_t_helper(1, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.0456)
    assert isclose(hmm.alpha_t_helper(0, 0, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.24)
    assert isclose(hmm.alpha_t_helper(0, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.08)
    assert isclose(hmm.alpha_t_helper(1, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.0456)

    # test beta
    emission_seq = hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B', 'B'])
    cache = np.full([len(emission_seq), len(hmm.transition_probs)], -1.0)
    assert isclose(hmm.beta_t_helper(0, 0, emission_seq, cache), 0.0324)
    assert isclose(hmm.beta_t_helper(0, 1, emission_seq, cache), 0.0297)
    assert isclose(hmm.beta_t_helper(1, 0, emission_seq, cache), 0.09)
    assert isclose(hmm.beta_t_helper(1, 1, emission_seq, cache), 0.09)
    assert isclose(hmm.beta_t_helper(2, 0, emission_seq, cache), 0.3)
    assert isclose(hmm.beta_t_helper(2, 1, emission_seq, cache), 0.3)
    assert isclose(hmm.beta_t_helper(3, 0, emission_seq, cache), 1)
    assert isclose(hmm.beta_t_helper(3, 1, emission_seq, cache), 1)

    label_seq = ['R', 'W', 'B', 'B']
    emission_seq = hmm._get_emission_idx_seq_from_label_seq(label_seq)
    cache = np.full([len(emission_seq), len(hmm.transition_probs)], -1.0)
    prob = 0
    for i in range(0, len(hmm.transition_probs)):
        prob += hmm.beta_t_helper(0, i, emission_seq, cache) * hmm.initial_probs[i]*hmm.emission_probs[i][emission_seq[0]]
    assert isclose(prob, hmm.calc_prob_output_sequence(label_seq))

    print(hmm.get_likelihood(0, 'S1', label_seq))
    print(hmm.get_likelihood(0, 'S2', label_seq))
    print(hmm.get_likelihood(1, 'S1', label_seq))
    print(hmm.get_likelihood(1, 'S2', label_seq))
    print(hmm.get_likelihood(2, 'S1', label_seq))
    print(hmm.get_likelihood(2, 'S2', label_seq))
    print(hmm.get_likelihood(3, 'S1', label_seq))
    print(hmm.get_likelihood(3, 'S2', label_seq))
