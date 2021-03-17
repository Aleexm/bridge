# Bridge environments ♣️

This github contains an environment for both the bidding (gym-bridge) and playing out the cards (gym-play).
Installation is simple:
 - Navigate to the desired local directory where you wish to place the codebase.
 - Run git clone git@github.com:Aleexm/bridge.git
 - Enter the created directory
 - Run pip install -r requirements.txt
 - Run pip -e install gym-bridge
 - Run pip -e install gym-play
 - Run pip install tensorflow==1.5.0 if you want to train another bidding agent.

# Gym Play ♠

## List of features
- PlayEnv can fully simulate the playing of the cards, including tricks taken, scoring, IMP conversion, etc.
- Has an integrated POMCP solver, which you can play against interactively.

# Gym Bridge  ♥️

OpenAI Gym environment for Bridge Bidding, with an integrated Double-Dummy Solver courtesy of Bo Haglund.
Using the DDS, we can check quickly whether a contract was made.

## List of Features
- Is able to wholly handle the bidding.
- Controller delegates (joint) obervations and action selection to correct sub-agent, masking hidden information.
- Simple RandomAgent/PassingAgent/DoublingAgent correctly interact with the environment.
- Double Dummy Solver is fully integrated! We can now check whether a contract was made :)
  - Credit to anntzer (https://github.com/anntzer/redeal) for showing how to wrap DDS into Python.
  - Full credit for the DDS goes to Bo Haglund (http://privat.bahnhof.se/wb758135/).
- Duplicate bridge format (https://en.wikipedia.org/wiki/Duplicate_bridge) implemented, where a team of 2x2 players sits NS / EW at two tables, receiving equal deals.
- Scoring: IMP (https://www.bridgewebs.com/barnstaple/Tactics%20at%20Imps.htm) with Duplicate Bridge Format fully works.

## Environment overview 
- A hand is encoded as a 52-bitvector of the ordered set of cards, i.e. 2C<2D<...<AH<AS. A '1' indicates this card is held.
- The vulnerability is encoded as a 2-bitvector. First bit is NS vuln, second bit is EW vuln.
- The bidding history is encoded as a 318-bitvector, where a 1 in the i-th entry denotes that the i-th bid in the possible maximum bidding sequence is called, i.e. ```p-p-p|1C-p-p-d-p-p-r-p-p|1D-p-p-d-p-p-r-p-p|...|7N-p-p-d-p-p-r-p-p```. The final pass can be inferred. See https://arxiv.org/abs/1903.00900 for details.
- Observations in this environment are the State, which contains all global information. A Controller then filters the joint observation and passes the correct local information to the acting agent.
- At each timestep, one of the four agents makes a bid and the state is updated. After three consecutive passes, the bidding phase concludes (4 passes if no contract bid was ever made.).
- A Double Dummy Solver is used to approximate the tricks made by the declaring partnership.

<img src="https://i.imgur.com/DBwuRnX.png" height="600">
