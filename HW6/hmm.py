import math

import numpy as np

class HMM:
    def __init__(self, initial_probs, transition_probs, emission_probs, state_labels, emission_labels):
        # sanity check
        # no fitting, fixed markov model
        assert transition_probs is not None
        assert emission_probs is not None

        assert len(emission_probs) == len(transition_probs)
        assert math.isclose(sum(initial_probs), 1.0)
        for state_transfer_probs in transition_probs:
            assert len(state_transfer_probs) == len(transition_probs)
            assert math.isclose(sum(state_transfer_probs), 1.0)
        for state_emission_probs in emission_probs:
            assert len(state_emission_probs) == len(emission_probs[0])
            assert math.isclose(sum(state_emission_probs), 1.0)
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

    def alpha_t_helper(self, t, state_idx, emission_idx_seq, cache): #cache-> array of t x states
        assert t >= 0
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

def test_hmm():
    hmm = HMM(initial_probs=[0.8, 0.2]
              , transition_probs= [[0.6, 0.4], [0.3, 0.7]]
              , emission_probs = [[0.3, 0.4, 0.3], [0.4, 0.3, 0.3]]
              , state_labels = ['S1', 'S2']
              , emission_labels = ['R', 'W', 'B'])
    cache = np.full([3, len(hmm.transition_probs)], -1.0)
    assert math.isclose(hmm.alpha_t_helper(2, 0, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.0162)
    assert math.isclose(hmm.alpha_t_helper(2, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.01764)
    assert math.isclose(hmm.alpha_t_helper(1, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.0456)
    assert math.isclose(hmm.alpha_t_helper(0, 0, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.24)
    assert math.isclose(hmm.alpha_t_helper(0, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.08)
    assert math.isclose(hmm.alpha_t_helper(1, 1, hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B']), cache), 0.0456)

    # test beta
    emission_seq = hmm._get_emission_idx_seq_from_label_seq(['R', 'W', 'B', 'B'])
    cache = np.full([len(emission_seq), len(hmm.transition_probs)], -1.0)
    assert math.isclose(hmm.beta_t_helper(0, 0, emission_seq, cache), 0.0324)
    assert math.isclose(hmm.beta_t_helper(0, 1, emission_seq, cache), 0.0297)
    assert math.isclose(hmm.beta_t_helper(1, 0, emission_seq, cache), 0.09)
    assert math.isclose(hmm.beta_t_helper(1, 1, emission_seq, cache), 0.09)
    assert math.isclose(hmm.beta_t_helper(2, 0, emission_seq, cache), 0.3)
    assert math.isclose(hmm.beta_t_helper(2, 1, emission_seq, cache), 0.3)
    assert math.isclose(hmm.beta_t_helper(3, 0, emission_seq, cache), 1)
    assert math.isclose(hmm.beta_t_helper(3, 1, emission_seq, cache), 1)

    label_seq = ['R', 'W', 'B', 'B']
    emission_seq = hmm._get_emission_idx_seq_from_label_seq(label_seq)
    cache = np.full([len(emission_seq), len(hmm.transition_probs)], -1.0)
    prob = 0
    for i in range(0, len(hmm.transition_probs)):
        prob += hmm.beta_t_helper(0, i, emission_seq, cache) * hmm.initial_probs[i]*hmm.emission_probs[i][emission_seq[0]]
    assert math.isclose(prob, hmm.calc_prob_output_sequence(label_seq))
