{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 77,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "\n",
    "def make_context_vector(context, word_to_ix):\n",
    "    idxs = [word_to_ix[w] for w in context]\n",
    "    return torch.tensor(idxs, dtype=torch.long)\n",
    "\n",
    "context_size = 4  # 2 words to the left, 2 to the right\n",
    "EMDEDDING_DIM = 100\n",
    "\n",
    "raw_text = \"\"\"We are about to study the idea of a computational process.\n",
    "Computational processes are abstract beings that inhabit computers.\n",
    "As they evolve, processes manipulate other abstract things called data.\n",
    "The evolution of a process is directed by a pattern of rules\n",
    "called a program. People create programs to direct processes. In effect,\n",
    "we conjure the spirits of the computer with our spells.\"\"\".split()\n",
    "\n",
    "\n",
    "# By deriving a set from `raw_text`, we deduplicate the array\n",
    "vocab = set(raw_text)\n",
    "vocab_size = len(vocab)\n",
    "\n",
    "word_to_ix = {word:ix for ix, word in enumerate(vocab)}\n",
    "ix_to_word = {ix:word for ix, word in enumerate(vocab)}\n",
    "\n",
    "data = []\n",
    "for i in range(2, len(raw_text) - 2):\n",
    "    context = [raw_text[i - 2], raw_text[i - 1],\n",
    "               raw_text[i + 1], raw_text[i + 2]]\n",
    "    target = raw_text[i]\n",
    "    data.append((context, target))\n",
    "\n",
    "\n",
    "class CBOW(torch.nn.Module):\n",
    "    def __init__(self, vocab_size, embedding_dim):\n",
    "        super().__init__()\n",
    "\n",
    "        #out: 1 x emdedding_dim\n",
    "        self.embeddings = nn.Embedding(vocab_size, embedding_dim)\n",
    "        self.linear1 = nn.Linear(context_size*embedding_dim, 128)\n",
    "        self.activation_function1 = nn.ReLU()\n",
    "        \n",
    "        #out: 1 x vocab_size\n",
    "        self.linear2 = nn.Linear(128, vocab_size)\n",
    "        self.activation_function2 = nn.LogSoftmax(dim = 1)\n",
    "        \n",
    "\n",
    "    def forward(self, inputs):\n",
    "        embeds = self.embeddings(inputs)\n",
    "        #print(embeds.shape)\n",
    "        embeds=embeds.view(-1,400)\n",
    "        #print(embeds.shape)\n",
    "        out = self.linear1(embeds)\n",
    "        out = self.activation_function1(out)\n",
    "        out = self.linear2(out)\n",
    "        out = self.activation_function2(out)\n",
    "        return out\n",
    "\n",
    "    def get_word_emdedding(self, word):\n",
    "        word = torch.tensor([word_to_ix[word]])\n",
    "        return self.embeddings(word).view(1,-1)\n",
    "\n",
    "\n",
    "model = CBOW(vocab_size, EMDEDDING_DIM)\n",
    "\n",
    "loss_function = nn.NLLLoss()\n",
    "optimizer = torch.optim.SGD(model.parameters(), lr=0.001)\n",
    "\n",
    "#TRAINING\n",
    "for epoch in range(50):\n",
    "    total_loss = 0\n",
    "\n",
    "    for context, target in data:\n",
    "        context_vector = make_context_vector(context, word_to_ix)  \n",
    "\n",
    "        log_probs = model(context_vector)\n",
    "\n",
    "        total_loss += loss_function(log_probs, torch.tensor([word_to_ix[target]]))\n",
    "\n",
    "    #optimize at the end of each epoch\n",
    "    optimizer.zero_grad()\n",
    "    total_loss.backward()\n",
    "    optimizer.step()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 112,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Output : spirits\n"
     ]
    }
   ],
   "source": [
    "a=['conjure','the','of','the']\n",
    "b=['spirits']\n",
    "context_vector = make_context_vector(a, word_to_ix) \n",
    "output=model(context_vector)\n",
    "print(f\"Output : {ix_to_word[torch.argmax(output).item()]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
