// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F26
//
// TeenyTinyEmbedder

let filename = 'data/dickinson.txt'; // try 'data/shakespeare.txt' or your own .txt file!
let vocabSize = 1000; // how many tokens the tokenizer learns
let maxLength = 16; // the longest passage to train on (in tokens)
let embedSize = 64; // the size of each embedding
let batchSize = 32; // how many pairs for each training step

let corpus; // the training data
let tokenizer; // the BPE tokenizer
let data; // the whole text as a list of token ids
let embedder; // the model!

let steps = 0;
let losses = [];

// For finding similar passages of text to a query
let queryInput;
let lines = []; // every line of the text, with its token ids and embedding
let groups = {}; // the same lines, grouped by how many tokens they have
let results = [];

async function setup() {
  createCanvas(800, 420);
  textFont('monospace');

  // Load the text
  let raw = await fetch(filename);
  corpus = await raw.text();

  // Train the tokenizer (this takes a few seconds!)
  tokenizer = new BPE();
  tokenizer.train(clean(corpus), vocabSize);

  // Encode the corpus!
  data = tokenizer.encode(clean(corpus));

  // Wait for TensorFlow.js to be ready!
  await tf.ready();
  embedder = new TinyEmbedder(tokenizer.vocab.length, embedSize);

  // Get ready to search
  prepareLines();
  queryInput = createInput('The sea was angry that day').position(20, 190).size(300);
  // Search when you press enter
  queryInput.changed(search);
  search();
}

function draw() {
  background(255);
  if (!embedder) return;

  // Train!
  let { anchors, partners } = getBatch();
  let loss = embedder.train(anchors, partners);
  tf.dispose([anchors, partners]);
  losses.push(loss);
  steps++;

  // Every 100 steps, search again to see what the model has learned
  if (steps % 100 === 0) {
    search();
  }

  drawLosses(20, 40, width - 40, 120);
  drawResults(20, 230);
}

// Clean the text before it becomes tokens: lowercase letters only, with a space before every word
// 'Because I could not stop for Death –' becomes ' because i could not stop for death'
// An embedding model only reads, it never writes, so it doesn't need capital letters or punctuation.
// Without them, 'Death' and 'death,' are the same token. (TeenyTinyGPT uses the same bpe.js
// on the original text, because it has to write!)
function clean(text) {
  text = text.toLowerCase(); // 'Death' becomes 'death'
  text = text.replace(/[^a-z]+/g, ' '); // anything that isn't a letter becomes one space
  return ' ' + text.trim(); // and a space in front of the first word too
}

// A batch of pairs: random passages (the "anchors") and the passages that come right after them
// (the "partners"). Neighbors in the text should end up as neighbors in the embedding space!
function getBatch() {
  // Pick random lengths, so the model learns to compare short and long passages
  let anchorLength = floor(random(2, maxLength + 1));
  let partnerLength = floor(random(2, maxLength + 1));
  let anchors = [];
  let partners = [];
  for (let i = 0; i < batchSize; i++) {
    let start = floor(random(data.length - anchorLength - partnerLength));
    let middle = start + anchorLength;
    anchors.push(data.slice(start, middle));
    partners.push(data.slice(middle, middle + partnerLength));
  }
  // Show one pair every 100 steps
  if (steps % 100 === 0) {
    console.log(`step ${steps}: "${tokenizer.decode(anchors[0])}" → "${tokenizer.decode(partners[0])}"`);
  }
  // Turn the anchors and partners into tensors of shape [batch, length] with dtype int32
  return {
    anchors: tf.tensor2d(anchors, [batchSize, anchorLength], 'int32'),
    partners: tf.tensor2d(partners, [batchSize, partnerLength], 'int32'),
  };
}

// Get every line of the text ready to search
function prepareLines() {
  let seen = {};
  for (let line of corpus.split('\n')) {
    line = line.trim();
    let ids = tokenizer.encode(clean(line));
    // Skip very short lines and lines we've already seen
    if (ids.length >= 3 && seen[line] === undefined) {
      seen[line] = true;
      let entry = { text: line, ids: ids, embedding: [], score: 0 };
      lines.push(entry);
      // Group the lines by how many tokens they have
      if (groups[ids.length] === undefined) {
        groups[ids.length] = [];
      }
      groups[ids.length].push(entry);
    }
  }
}

// Embed every line and the search, then find the lines most similar to the search
function search() {
  // The model keeps changing, so embed every line again.
  // Lines with the same number of tokens fit in one batch!
  for (let length in groups) {
    let passages = [];
    for (let line of groups[length]) {
      passages.push(line.ids);
    }
    let vectors = embedder.embeddings(passages);
    for (let i = 0; i < vectors.length; i++) {
      groups[length][i].embedding = vectors[i];
    }
  }

  // Embed the search, and compare it to every line
  let query = embedder.embeddings([tokenizer.encode(clean(queryInput.value()))])[0];
  for (let line of lines) {
    line.score = dotProduct(query, line.embedding);
  }
  // Sort from most to least similar, and keep the top 10
  lines.sort((a, b) => b.score - a.score);
  results = lines.slice(0, 10);
}

// Multiply and add up two embeddings. They're length 1, so this is the cosine similarity!
function dotProduct(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    sum += a[i] * b[i];
  }
  return sum;
}

// Draw the loss over time
function drawLosses(x, y, w, h) {
  let maxLoss = max(losses);
  noStroke();
  fill(0);
  textSize(14);
  textAlign(LEFT, BOTTOM);
  let latest = nf(losses[losses.length - 1], 1, 2);
  text(`step ${steps}   loss ${latest}`, x, y - 10);

  stroke(200);
  noFill();
  rect(x, y, w, h);
  stroke(0, 120, 255);
  beginShape();
  for (let i = 0; i < losses.length; i++) {
    let px = map(i, 0, max(losses.length - 1, 1), x, x + w);
    let py = map(losses[i], 0, maxLoss, y + h, y);
    vertex(px, py);
  }
  endShape();
}

// Draw the search split into tokens, then the most similar lines
function drawResults(x, y) {
  noStroke();
  fill(0);
  textSize(14);
  textAlign(LEFT, TOP);
  let tokens = tokenizer.tokenize(clean(queryInput.value()));
  text(`tokens: [${tokens.join('][')}]`, x, y);
  for (let i = 0; i < results.length; i++) {
    fill(150);
    text(nf(results[i].score, 1, 2), x, y + 30 + i * 16);
    fill(0);
    text(results[i].text, x + 50, y + 30 + i * 16);
  }
}
