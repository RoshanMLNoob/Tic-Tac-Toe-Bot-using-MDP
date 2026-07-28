# Tic-Tac-Toe-Bot-using-MDP
This is a Tic Tac Toe playing bot made using Markov Decision Process with Value Iteration, it is not
the best one you will find but for my first time ill say it is pretty good, the problem with this is that 
this predicts the moves of its future without considering the opponents move, leading to bad training on few iterations.

To play you just have to change play to True, and in the "TicTacToe" class which is assigned to variable "mdp" it has various 
settings you can change many things likein which file and from which file to pick 
the model weights from, and if you set play to False it will start training it
and save it in whichever file the file="..." was set to, and if set to None it will just do nothing, and there is also
a tournament one which if you put out of comment might work, if it doesn't you can contact me via commenting or the Issues tab.
