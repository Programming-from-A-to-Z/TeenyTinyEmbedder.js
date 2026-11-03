// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F26
//
// Byte Pair Encoding (BPE) tokenizer
// Start with single characters, merge the most common token pairs

class BPE {
  constructor() {
    // All the tokens (id is index into array)
    this.vocab = [];
    // All the merges (in order!)
    this.merges = [];
    // Every word in the text, split into tokens and counted:
    // {tokens: [' ', 't', 'h', 'e'], count: 100}
    this.words = [];
  }

  // Split text into words, keeping everything
  // A word is any sequence of letters (with optional space in front)
  // anything that isn't a letter is its own word.
  splitWords(text) {
    return text.match(/ ?[A-Za-z]+|[^A-Za-z]/g);
  }

  // Learn merges from a text until there are vocabSize tokens
  train(text, vocabSize) {
    this.start(text);
    while (this.vocab.length < vocabSize) {
      this.learnMerge();
    }
  }

  // Count all the word frequencies! (Simple text concordance) and set up the starting vocabulary
  start(text) {
    // Count every word, so each different word only has to be looked at once
    let counts = {};
    for (let word of this.splitWords(text)) {
      if (counts[word] === undefined) {
        counts[word] = 0;
      }
      counts[word]++;
    }

    // Every word starts out as single characters: ' the' is [' ', 't', 'h', 'e']
    for (let word in counts) {
      this.words.push({ tokens: word.split(''), count: counts[word] });
    }

    // The starting vocabulary is every character in the text
    this.vocab = Array.from(new Set(text));
    this.vocab.sort();
  }

  // Learn one new token
  learnMerge() {
    // Find the two tokens that appear next to each other most often
    let pair = this.mostCommonPair();
    // Together they become a new token!
    this.merges.push(pair);
    this.vocab.push(pair[0] + pair[1]);
    // Re-tokenize every word with the new token
    for (let word of this.words) {
      word.tokens = this.merge(word.tokens, pair);
    }
  }

  // Count every pair of tokens that sit next to each other
  countPairs() {
    let counts = {};
    for (let word of this.words) {
      for (let i = 0; i < word.tokens.length - 1; i++) {
        // The pair 't' then 'h' gets the key 't,h'
        let key = word.tokens[i] + ',' + word.tokens[i + 1];
        if (counts[key] === undefined) {
          counts[key] = 0;
        }
        // A word that appears N times counts the pair N times
        counts[key] += word.count;
      }
    }
    return counts;
  }

  // Find the most common pair of tokens!
  mostCommonPair() {
    let counts = this.countPairs();
    let best = '';
    let bestCount = 0;
    for (let key in counts) {
      if (counts[key] > bestCount) {
        best = key;
        bestCount = counts[key];
      }
    }
    return best.split(',');
  }

  // Replace any token pairs with the new merged token
  merge(tokens, pair) {
    let merged = [];
    let i = 0;
    while (i < tokens.length) {
      if (tokens[i] === pair[0] && tokens[i + 1] === pair[1]) {
        merged.push(pair[0] + pair[1]);
        i += 2;
      } else {
        merged.push(tokens[i]);
        i += 1;
      }
    }
    return merged;
  }

  // Split one word into tokens by repeating every merge, in the order they were learned
  // This is inefficient when you are tokenizing large blocks of text with repeated words.
  // You could remember each word that has been tokenized in a "cache"
  tokenizeWord(word) {
    let tokens = word.split('');
    for (let pair of this.merges) {
      tokens = this.merge(tokens, pair);
    }
    return tokens;
  }

  // Turn new text into tokens!
  tokenize(text) {
    let tokens = [];
    for (let word of this.splitWords(text)) {
      for (let token of this.tokenizeWord(word)) {
        tokens.push(token);
      }
    }
    return tokens;
  }

  // Turn text into token ids!
  encode(text) {
    let ids = [];
    for (let token of this.tokenize(text)) {
      ids.push(this.vocab.indexOf(token));
    }
    return ids;
  }

  // Turn token ids back into text!
  decode(ids) {
    let text = '';
    for (let id of ids) {
      text += this.vocab[id];
    }
    return text;
  }
}
