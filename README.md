# Multilayer Perceptron (MLP) from Scratch

Uma implementação do zero de uma rede neural **Multilayer Perceptron (MLP)** em Python utilizando apenas **NumPy**, desenvolvida para fins didáticos e para entender os fundamentos de Deep Learning sem dependência de frameworks de alto nível (como PyTorch ou TensorFlow).

---

## 🚀 Estado Atual da implementação

- [x] **Classe `Layer`**: Representação de uma camada da rede neural.
- [x] **Inicialização de Pesos**: Suporte às estratégias **He** e **Xavier** (Glorot).
- [x] **Forward Pass**: Propagação direta dos dados através da camada.
- [x] **Funções de Ativação**:
  - `ReLU` (Rectified Linear Unit)
  - `Sigmoid`
  - `Softmax`
- [x] **Backward Pass (Backpropagation)**
- [ ] **Funções de Perda (Loss Functions)** *(Em desenvolvimento)*
- [ ] **Otimizadores (SGD / Adam)** *(Em desenvolvimento)*

---

## 🛠️ Tecnologias Utilizadas

- **Python**
- **NumPy**: Para manipulação matricial e operações de álgebra linear.

---

## 📂 Estrutura do Projeto

```text
.
├── src/                # Código-fonte da rede neural
│   ├── layer.py        # Classe Layer (forward pass e inicializações)
│   └── activations.py  # Funções de ativação (ReLU, Sigmoid, Softmax)
├── requirements.txt    # Dependências do projeto
└── README.md           # Documentação
