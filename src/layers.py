from keras.engine.topology import Layer
import keras.backend as K
from keras.initializers import RandomUniform


class GRI(Layer):
    # NOTE: Not used
    def __init__(self, root_node, nodes, edges, **kwargs):
        """
        :param root_node: node on top of the hierarchy
        :param nodes: list of nodes (gene symbols) according to which the output nodes will be sorted
        :param edges: dictionary of edges (keys: *from* node, values: dictionary with key *to* node and
               value regulation type)
        """
        self._output_dim = len(nodes)
        self._root_node = root_node
        self._nodes = nodes
        self._edges = edges
        self._r_edges = self._reverse_edges(edges)
        self._structure = None
        super(GRI, self).__init__(**kwargs)

    @staticmethod
    def _reverse_edges(edges):
        r_edges = {}
        for tf, tgs in edges.items():
            for tg, reg_type in tgs.items():
                if tg in r_edges:
                    if tf not in r_edges[tg]:
                        r_edges[tg][tf] = reg_type
                else:
                    r_edges[tg] = {tf: reg_type}
        return r_edges

    def _create_params(self, node, latent_dim, nb_incoming):
        bias = self.add_weight(name='{}_bias'.format(self._root_node),
                               shape=(1, 1),
                               initializer='zeros',
                               trainable=True)
        weights = self.add_weight(name='{}_weights'.format(node),
                                  shape=(latent_dim + nb_incoming, 1),
                                  initializer='glorot_uniform',
                                  trainable=True)
        params = K.concatenate([bias, weights], axis=0)  # Shape=(1 + latent_dim + nb_incoming, 1)
        return params

    def build(self, input_shape):
        batch_size, latent_dim = input_shape

        # Structure is a list of tuples with format: (NODE, INCOMING_NODES, WEIGHTS)
        # The list is sorted so that node i does not depend on node j if j>i
        w = self._create_params(self._root_node, latent_dim, 0)
        self._structure = [(self._root_node, [], w)]

        nodes = set(self._nodes) - {self._root_node}
        remaining = len(nodes)
        while remaining > 0:
            for node in nodes:
                regulated_by = self._r_edges[node].keys()
                if all([parent not in nodes for parent in regulated_by]):
                    w = self._create_params(node, latent_dim, len(regulated_by))
                    t = (node, regulated_by, w)
                    self._structure.append(t)
                    nodes = nodes - {node}
            assert len(nodes) < remaining
            remaining = len(nodes)

        super(GRI, self).build(input_shape)  # Be sure to call this at the end

    def call(self, z, **kwargs):
        """
        :param z: Noise tensor. Shape=(batch_size, LATENT_DIM)
        :return: synthetic gene expressions tensor
        """
        # Dictionary holding the current value of each node in the GRI.
        # Key: gene symbol. Value: gene Keras tensor
        units = {}

        # Compute feedforward pass for GRI layer
        x_bias = K.ones_like(z)  # Shape=(batch_size, latent_dim)
        x_bias = K.mean(x_bias, axis=-1)[:, None]  # Shape=(batch_size, 1)
        for t in self._structure:
            node, incoming, weights = t
            x_units = [units[in_node] for in_node in incoming]
            x_in = K.concatenate([x_bias, z] + x_units, axis=-1)  # Shape=(batch_size, 1 + latent_dim + nb_incoming)
            x_out = K.dot(x_in, weights)  # Shape=(batch_size, 1)
            units[node] = K.tanh(x_out)

        return K.concatenate([units[node] for node in self._nodes], axis=-1)  # Shape=(batch_size, nb_nodes)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self._output_dim


class ClipWeights:
    """
    Keras constraint. Clips weights between two values
    """
    def __init__(self, min_value, max_value):
        self._min_value = min_value
        self._max_value = max_value

    def __call__(self, w):
        return K.clip(w, self._min_value, self._max_value)

    def get_config(self):
        return {'min_value': self._min_value,
                'max_value': self._max_value}


class NormWeights:
    """
    Keras constraint. Rescales weights so that the L1 norm is total_weights
    """
    def __init__(self, total_weights=10):
        self._total_weights = total_weights

    def __call__(self, w):
        w = self._total_weights * w / K.sum(K.abs(w))
        return w

    def get_config(self):
        return {'total_weights': self._total_weights}


class GeneWiseNoise(Layer):
    """
    Keras layer. Adds learnable Gaussian white noise
    """
    def __init__(self, noise_rate=0.25, **kwargs):
        self._w = None
        self._noise_rate = noise_rate
        super(GeneWiseNoise, self).__init__(**kwargs)

    def build(self, input_shape):
        nb_genes = input_shape[1]
        self._w = self.add_weight(name='w',
                                  shape=(nb_genes,),
                                  initializer='uniform',
                                  trainable=True,
                                  constraint=NormWeights(total_weights=self._noise_rate*nb_genes)
                                  )
        super(GeneWiseNoise, self).build(input_shape)

    def call(self, x, **kwargs):
        noise = K.random_normal(K.shape(x), mean=0, stddev=1)
        additive_noise = noise * self._w
        out = x + additive_noise
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_weights_norm(self):
        return K.sum(K.abs(self._w))


class MinibatchDiscrimination(Layer):
    def __init__(self, units=5, units_out=10, **kwargs):
        self._w = None
        self._units = units
        self._units_out = units_out
        super(MinibatchDiscrimination, self).__init__(**kwargs)

    def build(self, input_shape):
        self._w = self.add_weight(name='w',
                                  shape=(input_shape[1], self._units * self._units_out),
                                  initializer='uniform',
                                  trainable=True
                                  )
        super(MinibatchDiscrimination, self).build(input_shape)

    def call(self, x, **kwargs):
        h = K.dot(x, self._w)  # Shape=(batch_size, units * units_out)
        h = K.reshape(h, (-1, self._units, self._units_out))  # Shape=(batch_size, units, units_out)
        h_t = K.permute_dimensions(h, [1, 2, 0])  # Shape=(units, units_out, batch_size)
        diffs = h[..., None] - h_t[None, ...]  # Shape=(batch_size, units, units_out, batch_size)
        abs_diffs = K.sum(K.abs(diffs), axis=1)  # Shape=(batch_size, units_out, batch_size)
        features = K.sum(K.exp(-abs_diffs), axis=-1)  # Shape=(batch_size, units_out)
        return K.concatenate([x, features])

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1] + self._units_out