# Proactive Agent Behaviour in Dynamic Distributed Constraint Optimisation Problems.
This is the codes accompanying the paper with the title above. 
There are three parts to this project:

## The Robocup Rescue Simulation Server
The RCRS is the simulation environment used for evaluating 
the proposed approach in the study.
Follow [the official repo](https://github.com/roborescue/rcrs-server) to set it up
locally.

## Agent Communication
The communication layer used by the agents in the experiments rely on the
AMQP RabbitMQ broker. Follow the [official installation guide](https://www.rabbitmq.com/docs/download)
to set the broker up locally. We found it easier to use the Docker image for our experiments. 
We used the [Python client library (Pika)](https://www.rabbitmq.com/tutorials/tutorial-one-python)
in this repo.

### The Agents Implementation
The dependencies of this project can be installed by running:
```bash
$ pip install -r requirements.txt
```
Once the RCRS server and RabbitMQ broker have been started, 
the agents program in this codebase can be started by running:
```bash
$ python launch.py -fb 12 -fs 1
```
The above command starts 12 fire brigade agents and 1 fire station.

See `rcrs_ddcop.core.fireBrigadeAgent` for the fire brigade agent 
program. Each fire brigade agent composes a `core.bdi_agent.BDIAgent`
program that is in charge of spinnig up the DCOP algorithm, the dynamic
graph algorithm, the information sharing algorithm, experience buffer, and other components
used for agent reasoning.

The `algorithms.dcop.DCOP` class is the parent class of the DCOP algorithms used in the experiments.
It creates the `XGBoostTrainer` which trains the local agent model on the shared experiences in
the buffer.