import socket

from kaggle_environments import make


def patched_connect(*args, **kwargs):
    # It depends on your testing purpose
    # You may want a exception, add here
    # If you test unconnectable situations
    # it can stay like this
    print("socket.connect() is disabled")


socket.socket.connect = patched_connect


def simple_verbose_agent1(obs, _):
    # if agent is guesser and turnType is "ask"
    if obs.turnType == "ask":
        response = "Is it a duck?"
    # if agent is guesser and turnType is "guess"
    elif obs.turnType == "guess":
        response = "duck"
    # if agent is the answerer
    elif obs.turnType == "answer":
        response = "no"
    return response


if __name__ == "__main__":
    logs = []
    env = make("llm_20_questions", debug=True, logs=logs)
    game_output = env.run(
        agents=[
            simple_verbose_agent1,
            simple_verbose_agent1,
            simple_verbose_agent1,
            "/kaggle_simulations/agent/main.py",
        ]
    )
    print("Game output: ", game_output)
    print("Logs: ", logs)
