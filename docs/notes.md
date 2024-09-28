# Reinforcement Learning to Embedding Pipeline
- Training dataset is generated from RL trained policy:
    - Saved as a pickle file:
    - A list of n amount of training simulations
        - In each index of list there is a dictionary:
            - Frame number: int
            - input_prompt:
                - I'm observing 1 cars and 0 pedestrians.
A moving car; Angle in degrees: -0.00; Distance: 7.80m; Direction of travel: same direction as me; My attention: 100%
The distance to the closest intersection is 30.32m
There is no traffic lights.
My car 0.00m from the lane center and 0.00 degrees right off center.
My current speed is 0.00 mph
Steering wheel is 0.00% right.
I need to go straight for at least 60.32m.

                    Here are my actions:

                    - Accelerator pedal 2%
                    - Brake pedal 0%
                    - Going to steer 7% to the left.


- Input prompt: This is the lanGen(v_car, vped, vego, vroute, response of rl) they do this because: The inclusion of orl is optional, and we generate two different versions of pseudo labels to cater
to different requirements: 
    1) Without Attention/Action: Employed during the representation pre-
    training stage (see Subsection 3.4.1), where the inference of attentions and actions is not required.
    2) With Attention/Action: Utilized for VQA labeling with GPT during the fine-tuning stage (see
    Subsection 3.4.2). This equips GPT with the ability to ask specific questions about attentions and
    actions, thereby empowering the driving LLM agent with the ability to reason about attentions and
    actions.