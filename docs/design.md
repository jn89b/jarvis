# Design Architecture
- Abstract Environment
    - Environment will spawn the agents and battlespace
    - Environment will set the agents into the battlespace
    - Has a battle space 
        - Battle space will have a list of agents
            - Each agents will have an instance of JSBSIM

- When we call out step from gymnasium environment
    - Battlespace will step through each of the agents commands using aircraftsim API


## Idiot Checks
- Make sure each aircraft can move 
- Make sure agent is mapped with correct actions 
