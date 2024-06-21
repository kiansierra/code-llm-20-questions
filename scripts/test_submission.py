from kaggle_environments import make

def simple_verbose_agent1(obs, cfg):
    
    # if agent is guesser and turnType is "ask"
    if obs.turnType == "ask":
        response = "Is it a duck?"
    # if agent is guesser and turnType is "guess"
    elif obs.turnType == "guess":
        response = "duck"
    # if agent is the answerer
    elif obs.turnType == "answer":
        response = "no"
        
        
if __name__ == "__main__":
    env = make("llm_20_questions")
    game_output = env.run(agents=[simple_verbose_agent1,simple_verbose_agent1, simple_verbose_agent1, "/subs/main.py"])
    print("Game output: ", game_output)