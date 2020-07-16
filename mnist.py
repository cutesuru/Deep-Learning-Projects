{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.nn.functional as F\n",
    "import torchvision\n",
    "from torchvision import datasets,transforms"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "train=datasets.MNIST(\"\",train=True,download=True,transform=transforms.Compose([transforms.ToTensor()]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "test=datasets.MNIST(\"\",train=False,download=True,transform=transforms.Compose([transforms.ToTensor()]))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "trainset = torch.utils.data.DataLoader(train,batch_size=32,shuffle=True)\n",
    "testset=torch.utils.data.DataLoader(test,batch_size=32,shuffle=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "net(\n",
      "  (fc1): Linear(in_features=784, out_features=64, bias=True)\n",
      "  (fc2): Linear(in_features=64, out_features=64, bias=True)\n",
      "  (fc3): Linear(in_features=64, out_features=64, bias=True)\n",
      "  (fc4): Linear(in_features=64, out_features=10, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "class net(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.fc1=nn.Linear(784,64)\n",
    "        self.fc2=nn.Linear(64,64)\n",
    "        self.fc3=nn.Linear(64,64)\n",
    "        self.fc4=nn.Linear(64,10)\n",
    "    \n",
    "    def forward(self,x):\n",
    "        x=F.relu(self.fc1(x))\n",
    "        x=F.relu(self.fc2(x))\n",
    "        x=F.relu(self.fc3(x))\n",
    "        x=self.fc4(x)\n",
    "        \n",
    "        return F.log_softmax(x,dim=1)\n",
    "        \n",
    "\n",
    "n=net()\n",
    "print(n)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor(0.1831, grad_fn=<NllLossBackward>)\n",
      "tensor(1.3411e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(3.3228e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(3.6135e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(5.0291e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(6.3702e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(5.8356e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0.0028, grad_fn=<NllLossBackward>)\n",
      "tensor(4.2096e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(5.0664e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(6.3330e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0032, grad_fn=<NllLossBackward>)\n",
      "tensor(9.4994e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(4.2018e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0091, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0.2822, grad_fn=<NllLossBackward>)\n",
      "tensor(0.3738, grad_fn=<NllLossBackward>)\n",
      "tensor(3.6135e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(1.0729e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(2.6115e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(1.9110e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(1.7136e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(1.6130e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(1.3746e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(9.9464e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(2.5517e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0011, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0345, grad_fn=<NllLossBackward>)\n",
      "tensor(3.7253e-09, grad_fn=<NllLossBackward>)\n",
      "tensor(1.1176e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(6.8017e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(1.0431e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0002, grad_fn=<NllLossBackward>)\n",
      "tensor(6.3835e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(4.8429e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0001, grad_fn=<NllLossBackward>)\n",
      "tensor(3.2744e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0010, grad_fn=<NllLossBackward>)\n",
      "tensor(1.4031e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(0.1889, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0.0004, grad_fn=<NllLossBackward>)\n",
      "tensor(3.0175e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(3.0547e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(1.1176e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(5.2154e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0098, grad_fn=<NllLossBackward>)\n",
      "tensor(1.8030e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0002, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(8.9407e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(5.7551e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(1.5050e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(2.9732e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(7.4506e-09, grad_fn=<NllLossBackward>)\n",
      "tensor(5.4582e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(2.1979e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(7.4506e-09, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(5.0664e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(4.2693e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(7.4506e-09, grad_fn=<NllLossBackward>)\n",
      "tensor(2.6077e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(1.4901e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(1.3038e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(0., grad_fn=<NllLossBackward>)\n",
      "tensor(9.1778e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(1.1176e-07, grad_fn=<NllLossBackward>)\n",
      "tensor(0.0009, grad_fn=<NllLossBackward>)\n",
      "tensor(7.2490e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(4.6414e-06, grad_fn=<NllLossBackward>)\n",
      "tensor(1.8626e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(5.8886e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(7.8231e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(1.8228e-05, grad_fn=<NllLossBackward>)\n",
      "tensor(6.7055e-08, grad_fn=<NllLossBackward>)\n",
      "tensor(1.0734e-05, grad_fn=<NllLossBackward>)\n"
     ]
    }
   ],
   "source": [
    "import torch.optim as optim\n",
    "optimizer=optim.Adam(n.parameters(),lr=0.001)\n",
    "epochs=100\n",
    "for i in range(epochs):\n",
    "    for data in trainset:\n",
    "        x,y=data\n",
    "        n.zero_grad()\n",
    "        output=n(x.view(-1,784))\n",
    "        loss=F.nll_loss(output,y)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "    print(loss)    \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Accuracy:0.062\n"
     ]
    }
   ],
   "source": [
    "correct=0\n",
    "total=0\n",
    "with torch.no_grad():\n",
    "    for data in trainset:\n",
    "        x,y=data\n",
    "        output=n(x.view(-1,784))\n",
    "        for idx,i in enumerate(output):\n",
    "            if(y[idx]==torch.argmax(i)):\n",
    "                correct+=1\n",
    "            total+=1    \n",
    "print(f\"Accuracy:{round(correct/total,3)}\")            \n",
    "                 \n",
    "        \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.imshow(x[0].view(28,28))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tensor(4)\n"
     ]
    }
   ],
   "source": [
    "print(y[0])"
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
