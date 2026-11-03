// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F26
//
// TeenyTinyEmbedder

class TinyEmbedder {
  constructor(vocabSize, embedSize) {
    this.embedSize = embedSize;

    // 1. EMBEDDINGS
    // A vector for every token in the vocabulary
    // (No position embedding! See the README for why.)
    this.tokenEmbedding = tf.layers.embedding({ inputDim: vocabSize, outputDim: embedSize });

    // 2. ATTENTION
    // Three dense layers, one for a query, a key, and a value
    this.query = tf.layers.dense({ units: embedSize, useBias: false });
    this.key = tf.layers.dense({ units: embedSize, useBias: false });
    this.value = tf.layers.dense({ units: embedSize, useBias: false });

    // 3. MULTILAYER PERCEPTRON
    this.hidden = tf.layers.dense({ units: embedSize * 4, activation: 'relu' });
    this.mlpOutput = tf.layers.dense({ units: embedSize });

    this.optimizer = tf.train.adam(0.003);

    // Run a prediction once so all the weights get created
    tf.tidy(() => this.predict(tf.zeros([1, 1], 'int32')));
  }

  // Every token looks at every other token and mixes in what it finds
  attention(x) {
    let q = this.query.apply(x);
    let k = this.key.apply(x);
    let v = this.value.apply(x);
    let scores = tf.matMul(q, k, false, true); // [batch, length, length]
    scores = scores.div(Math.sqrt(this.embedSize));
    // No mask! This model reads, it doesn't write, so it can look forward and backward
    let weights = tf.softmax(scores);
    return tf.matMul(weights, v);
  }

  // Token ids [batch, length] -> one embedding per passage [batch, embedSize]
  predict(inputs) {
    let x = this.tokenEmbedding.apply(inputs);
    // Attention and the MLP (adding for "residual connections"), just like TeenyTinyGPT
    x = x.add(this.attention(x));
    x = x.add(this.mlpOutput.apply(this.hidden.apply(x)));
    // Average all the tokens' vectors into one vector: "mean pooling"
    let average = x.mean(1); // [batch, embedSize]
    // Scale every vector to a length of 1, so a dot product is the cosine similarity
    return average.div(average.norm('euclidean', 1, true));
  }

  // One step of training: each anchor should be most similar to its own partner
  train(anchors, partners) {
    let loss = this.optimizer.minimize(() => {
      let a = this.predict(anchors); // [batch, embedSize]
      let b = this.predict(partners); // [batch, embedSize]
      // Compare every anchor with every partner: a [batch, batch] grid of cosine similarities
      let similarity = tf.matMul(a, b, false, true);
      // Anchor 0 goes with partner 0, anchor 1 with partner 1, and so on.
      // The right answers are the diagonal of the grid: the identity matrix!
      let answers = tf.eye(anchors.shape[0]);
      // Cosine similarity only goes from -1 to 1, multiplying by 20 makes softmax more decisive
      return tf.losses.softmaxCrossEntropy(answers, similarity.mul(20));
    }, true);
    let value = loss.arraySync();
    loss.dispose();
    return value;
  }

  // A helper function: lists of token ids (all the same length!) -> embeddings as plain JS arrays
  embeddings(passages) {
    return tf.tidy(() => {
      let inputs = tf.tensor2d(passages, [passages.length, passages[0].length], 'int32');
      return this.predict(inputs).arraySync();
    });
  }
}
