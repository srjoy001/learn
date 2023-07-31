import flwr as fl
import sys


# Start Flower server for three rounds of federated learning
fl.server.start_server(
    server_address='localhost:'+str(sys.argv[1]),
    config=fl.server.ServerConfig(num_rounds=10),
    grpc_max_message_length=1024*1024*1024,

)
