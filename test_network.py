from othello import Othello
from networks import ResNet
import torch

import matplotlib.pyplot as plt

game = Othello()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

state = game.get_initial_state()
# state = game.get_next_state(state, 2, -1)
# state = game.get_next_state(state, 4, -1)
# state = game.get_next_state(state, 6, 1)
# state = game.get_next_state(state, 8, 1)


encoded_state = game.get_encoded_state(state)

tensor_state = torch.tensor(encoded_state, device=device).unsqueeze(0)

model = ResNet(game, 4, 64, device=device)
# model.load_state_dict(torch.load('model_2.pt', map_location=device))
model.eval()

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value)

print(state)
print(tensor_state)

plt.bar(range(game.action_size), policy)
plt.show()