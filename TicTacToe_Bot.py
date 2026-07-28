import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

class TicTacToe(object):


    def save_model(self):
        Dict = self.V
        with open(self.file , "wb") as f:
            pickle.dump(Dict , f)

    def load_model(self):
        try:
            with open(self.file , "rb") as f:
                data = pickle.load(f)
                # Use .get to ensure it handles both old and new formats
                self.V = data.get("V" , {})
                print(bool({}) , len(self.V))
        except Exception as e:
            print(f"Error loading model: {e}")
            self.V = {}

    def __init__(self , s0=np.array([[0,0,0],[0,0,0],[0,0,0]]) , alpha:float=0.1 , gamma:float=0.9 , file=r"C:\Users\rosha\OneDrive\Desktop\Python_ML\Machine_Learning_Testing_Tweaking\Testing\Bot\data.pkl"):

        self.s0 = s0
        self.s = s0
        self.rew = 1
        self.file = file
        self.load_model()


        self.alpha = alpha
        self.gamma = gamma

    def _softmax(self , distribution:list)->np.array:
        dis = np.array(distribution)
        return np.exp(dis)/np.sum(np.exp(dis))

    def _actions(self , State=None):
        if State is None:
            State = self.s
        rows , cols = np.where(State == 0)
        return list(zip( rows , cols ))
    
    def _win(self , State=None):
        if State is None:
            State=self.s

        d , d0 = State.diagonal() , np.fliplr(State).diagonal()

        if (d==d[0]).all() and d[0] != 0:
            return d[0]
        elif (d0==d0[0]).all() and d0[0] != 0:
            return d0[0]
        for rows , cols in zip(State , State.T):
            if (rows==rows[0]).all() and rows[0] != 0:
                return rows[0]
            elif (cols==cols[0]).all() and cols[0] != 0:
                return cols[0]

        return 0
    
    def _represent(self):
        grid = self.s.copy().astype(str)
        grid[grid=="0"]=' '
        grid[grid=="1"]='X'
        grid[grid=="-1"]='O'
        return grid

    def transition(self , action:tuple , player:int=1 , play:bool=False , State=None)->np.array:
        if State is None:
            if play is True:
                self.s[*action] = player
                return self.s
            else:
                state = self.s.copy()
                state[*action] = player
                return state
        else:
            State[*action] == player
            return State
    
    def R(self , State=None):

        if State is None:
            State = self.s

        if self._win(State=State) == 1:
            return 10 #Reward for winning
        elif self._win(State=State) == -1:
            return -10 #punishment 
        else:
            if self._actions() == [()] or self._actions == [] or self._actions == ():
                return 9 #avg for draw
            else:
                return -1 #bad at each move
            
    def update_V(self , next_state , next_state2 , k=False):
        
        state0 = tuple(self.s.flatten())
        state1 = tuple(next_state.flatten())
        state2 = tuple(next_state2.flatten())

        r1 , r2 = self.R(next_state) , self.R(next_state2)
        if k is True:
            print(self.V.get(state0 , 0) , self.V.get(state1 , 0) , self.V.get(state2 , 0))

        target = r1 + (self.gamma*r2) + ((self.gamma**2) * self.V.get(state2 , 0))

        error = target - self.V.get(state0 , 0)
        self.V[state0] = self.V.get(state0 , 0) + (self.alpha * error)

        return True 

    def policy(self , State=None , p:int=1 , epsilon=0.25):
        if State is None:
            vals = []
            act = self._actions()
            for a in act:
                vals.append(self.V.get(tuple(self.transition(action=a , player=p , play=False).flatten()) , 0))
            self.probabilities = self._softmax(vals)

            if np.random.rand() < epsilon or self.s.all()==self.s0.all():
                action = act[np.random.choice(len(act))]
            else:
                action = act[np.argmax(self.probabilities)]

            self.transition(action=action , player=1 , play=True)

            return action
        else:
            vals = []
            act = self._actions(State=State)
            for a in act:
                vals.append(self.V.get(tuple(self.transition(action=a , player=p , play=False , State=State).flatten()) , 0))
            self.probabilities = self._softmax(vals)

            if np.random.rand() < epsilon or self.s.all()==self.s0.all():
                action = act[np.random.choice(len(act))]
            else:
                action = act[np.argmax(self.probabilities)]

            return action

    def Play(self , rounds , player_goes_first=False):

        for _ in range(rounds):
            self.s = self.s0
            self.s = np.array([[0,0,0],
                               [0,0,0],
                               [0,0,0]])
            while True:

                if self._win() != 0:
                    break
                elif self._actions() == [()] or not self._actions():
                    break

                move_bot = self.policy()
                print(self._represent())

                if self._win() != 0:
                    break
                elif self._actions() == [()] or not self._actions():
                    break

                move = tuple(input("Move as a Tuple: ").split(","))
                move = (int(move[0])-1 , int(move[1])-1)

                valid_moves = self._actions()
                if move not in valid_moves:
                    print("Invalid Move")
                    choice = np.random.choice(range(len(valid_moves)))
                    move = valid_moves[choice]
                print("\n")
                self.transition(action=move , player=-1 , play=True)
                print(self._represent())

                #Parameter Update
                s1 = self.transition(action=move_bot)
                s2 = self.transition(action=self.policy(State=s1) , State=s1)

                self.update_V(next_state=s1 , next_state2=s2)
                #print(self.V.get(tuple(self.s.flatten())))#

                if self._win() != 0:
                    break
                elif self._actions() == [()] or not self._actions():
                    break
            result = self._win()
            if result == 1:
                print("You Lost !")
            elif result == -1:
                print("You Won !")
            else:
                print("It's a Draw !")
        self.save_model()

    def Train(self, rounds): #Made by AI
        # To track progress every 10 rounds
        #plt.ion()
        running_rewards = []
        plotting_obj = []
        summ_rewards = []

        for i in range(1, rounds + 1):
            self.s = np.zeros((3, 3))
            game_reward = 0
        
            while self._win() == 0 and len(self._actions()) > 0:
                # 1. Player 1's turn
                move1 = self.policy(p=1)
                s1 = self.transition(action=move1, player=1, play=False)
            
                # Check for terminal state
                r1 = self.R(s1)
                game_reward += r1
                if self._win(State=s1) != 0 or len(self._actions(State=s1)) == 0:
                    break
                
                # 2. Player 2's turn
                move2 = self.policy(p=-1)
                s2 = self.transition(action=move2, player=-1, play=False)
            
                # 3. Update V
                self.update_V(next_state=s1, next_state2=s2)
            
                # 4. Advance board
                self.transition(action=move1, player=1, play=True)
                self.transition(action=move2, player=-1, play=True)
            
                if self._win() != 0: break
        
            running_rewards.append(game_reward)
            summ_rewards.append(np.sum((running_rewards)))
        
            # Print progress every 10 rounds
            o = 75
            o1 = 1
            if i % o == 0:
                avg_reward = np.mean(running_rewards[-o:])
                plotting_obj.append(len(self.V))
                print(f"Round {i}: Average Reward over last {o} games = {avg_reward:.2f} | Observed States: {plotting_obj[-1]}")

                #plt.clf()

                plt.plot(list(range(1,len(plotting_obj)+1))[::o1] , plotting_obj[::o1] , label="Number of States")
                plt.plot(list(range(1,len(summ_rewards)+1))[::o1] , summ_rewards[::o1] , label="Sum of Rewards")
                plt.plot(list(range(1,len(running_rewards)+1))[::o1] , running_rewards[::o1] , label="Running Rewards")

                #plt.legend()
                #plt.show()
                #plt.pause(0.01)
        
        plt.plot(list(range(1,len(plotting_obj)+1))[::o1] , plotting_obj[::o1] , label="Number of States")
        plt.plot(list(range(1,len(summ_rewards)+1))[::o1] , summ_rewards[::o1] , label="Sum of Rewards")
        plt.plot(list(range(1,len(running_rewards)+1))[::o1] , running_rewards[::o1] , label="Running Rewards")
        plt.legend()
        plt.show()
                

        self.save_model()
        print("Training complete.")

file1 = r"C:\Users\rosha\OneDrive\Desktop\Python_ML\Machine_Learning_Testing_Tweaking\Testing\Bot\data.pkl" #Human Trained Model
file2 = r"C:\Users\rosha\OneDrive\Desktop\Python_ML\Machine_Learning_Testing_Tweaking\Testing\Bot\data2.pkl" #Self Trained Model

mdp = TicTacToe(alpha=0.025 , gamma=0.95 , file=file2)

play = False
iterations = 50000

if play is True:
    print(mdp._represent())
    print("\n")
    mdp.Play(rounds=5)
elif play is None:
    pass
else:
    mdp.Train(rounds=iterations)


def tournament(model1_path, model2_path, rounds=10):
    
    game = TicTacToe()
    
    with open(model1_path, "rb") as f:
        brain1 = pickle.load(f)["V"]
    with open(model2_path, "rb") as f:
        brain2 = pickle.load(f)["V"]
        
    for r in range(1, rounds + 1):
        game.s = np.zeros((3, 3))
        turn = 1
        
        print(f"\n--- Starting Match {r} ---")
        
        while game._win() == 0 and len(game._actions()) > 0:
            # Switch brain based on player
            game.V = brain1 if turn == 1 else brain2
            
            move = game.policy(p=turn)
            game.transition(action=move, player=turn, play=True)
            
            # Optional: Print board after every move to watch the battle
            print(f"Player {turn} moves to {move}")
            print(game._represent())
            
            turn *= -1
            
        result = game._win()
        if result == 1:
            print(f"Match {r} Result: Model 1 (X) Won!")
        elif result == -1:
            print(f"Match {r} Result: Model 2 (O) Won!")
        else:
            print(f"Match {r} Result: It's a Draw!")

#tournament(file1 , file2 , 100)
