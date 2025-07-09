# 🔁 LSTM (Long Short-Term Memory) – Complete Notes with Diagrams

---

## 🧠 What is LSTM?

LSTM (Long Short-Term Memory) is a type of Recurrent Neural Network (RNN) designed to learn long-term dependencies. It solves problems standard RNNs face, like the **vanishing gradient problem**, by introducing **memory cells and gates**.

---

## ❌ Problems with Standard RNNs

### 1. Vanishing Gradient Problem
During backpropagation, gradients shrink exponentially → earlier layers learn very little.

![Vanishing Gradient](https://miro.medium.com/v2/resize:fit:640/format:webp/1*zE3WJ_E4XfvvKV4lik40Cw.png)

---

### 2. Short-Term Memory
RNNs forget long-term context over time. Input from far earlier in the sequence has little impact.

### 3. Exploding Gradients
Gradients can also grow uncontrollably, making training unstable (solved by gradient clipping).

---

## ✅ What LSTM Does Better

LSTM uses **gates** and a **cell state** to:

- Keep long-term memory (via cell state)
- Decide what to forget
- Decide what new info to store
- Decide what to output

---

## 🧱 LSTM Architecture Overview

![LSTM Cell](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

---

## 🔬 Step-by-Step Inside an LSTM Cell

---

### 🔷 1. Forget Gate (`fₜ`)

Determines which parts of the previous cell state `Cₜ₋₁` to forget.

![Forget Gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-f.png)

🧮 Formula:

![Forget Gate Equation](https://latex.codecogs.com/png.image?\dpi{150}&space;f_t%20=%20\sigma(W_f%20\cdot%20[h_{t-1},%20x_t]%20+%20b_f))

---

### 🔷 2. Input Gate (`iₜ` and `Ĉₜ`)

Decides which new information to add to the cell state.

![Input Gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-i.png)

🧮 Formula:

![Input Gate Equation](https://latex.codecogs.com/png.image?\dpi{150}&space;i_t%20=%20\sigma(W_i%20\cdot%20[h_{t-1},%20x_t]%20+%20b_i))

![Candidate State Equation](https://latex.codecogs.com/png.image?\dpi{150}&space;\tilde{C}_t%20=%20\tanh(W_C%20\cdot%20[h_{t-1},%20x_t]%20+%20b_C))

---

### 🔷 3. Cell State Update (`Cₜ`)

Update the cell state using the forget gate and input gate results.

![Cell State Update](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-C.png)

🧮 Formula:

![Cell State Update Equation](https://latex.codecogs.com/png.image?\dpi{150}&space;C_t%20=%20f_t%20*%20C_{t-1}%20+%20i_t%20*%20\tilde{C}_t)

---

### 🔷 4. Output Gate (`oₜ` and `hₜ`)

Determines what to output to the next time step.

![Output Gate](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-focus-o.png)

🧮 Formula:

![Output Gate Equation](https://latex.codecogs.com/png.image?\dpi{150}&space;o_t%20=%20\sigma(W_o%20\cdot%20[h_{t-1},%20x_t]%20+%20b_o))

![Hidden State Equation](https://latex.codecogs.com/png.image?\dpi{150}&space;h_t%20=%20o_t%20*%20\tanh(C_t))

---

## 🧠 Full View – LSTM Data Flow

![Full LSTM](https://colah.github.io/posts/2015-08-Understanding-LSTMs/img/LSTM3-chain.png)

---

## 📊 Applications of LSTM

| Domain         | Use Case                             |
|----------------|---------------------------------------|
| NLP            | Text generation, translation, chatbots |
| Time Series    | Stock prices, weather, IoT sensors    |
| Healthcare     | ECG/EEG, diagnosis prediction         |
| Speech         | Voice recognition, text-to-speech     |
| Recommendation | Session-based filtering               |

---

## 🛠️ Code Snippet – LSTM in Keras

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, Dense

model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=64, input_length=max_seq_len))
model.add(LSTM(100))
model.add(Dense(vocab_size, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

---

## 📌 Glossary

| Symbol           | Meaning                           |
|------------------|------------------------------------|
| `xₜ`             | Input at time step t               |
| `hₜ`             | Hidden state at time t             |
| `Cₜ`             | Cell state (long-term memory)      |
| `fₜ`             | Forget gate output                 |
| `iₜ`             | Input gate output                  |
| `oₜ`             | Output gate output                 |
| `W`, `b`         | Weights and biases                 |

---

## 🔄 LSTM vs Alternatives

| Model             | Features                          |
|-------------------|-----------------------------------|
| **Vanilla RNN**   | Simple, but forgets long-term info |
| **LSTM**          | Remembers long-term context        |
| **GRU**           | Simpler than LSTM, nearly as good  |
| **Transformer**   | Attention mechanism, parallelized  |

---

## 📝 References

- [Chris Olah’s Blog on LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Coursera: NLP Sequence Models](https://www.coursera.org/learn/nlp-sequence-models)
- [Keras LSTM Docs](https://keras.io/api/layers/recurrent_layers/lstm/)

---

## ✅ TL;DR

- RNNs struggle with long-term dependencies.
- LSTM solves this using gates and a memory cell.
- LSTM is widely used in NLP, time series, and sequence prediction tasks.

