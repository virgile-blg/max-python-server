import argparse
from pythonosc import osc_server, dispatcher, udp_client

from inference import Predictor

class OSCServer(object):
    def __init__(self, in_port:int=8000, out_port:int=8000, ip='127.0.0.1', osc_attributes=[], model_folder='models/shortres_msd',*args) -> None:
        """Initialize server

        Args:
            in_port (int): input port
            out_port (int): output port
            ip (str, optional): _description_. Defaults to '127.0.0.1'.
        """
        super(OSCServer, self).__init__()
        self.osc_attributes = osc_attributes
        self.model_folder = model_folder
        # OSC library objects
        self.dispatcher = dispatcher.Dispatcher()
        self.client = udp_client.SimpleUDPClient(ip, out_port) # Client properties
        print(f"Initialized OSC client sending to port {out_port}")
        # Bindings for server
        self.init_bindings(self.osc_attributes)
        self.server = osc_server.BlockingOSCUDPServer((ip, in_port), self.dispatcher) # Server properties
        print(f'Initialized OSC server listening to port {in_port}')
        self.in_port, self.out_port, self.ip = in_port, out_port, ip
        
        # Init ML Predictor
        self.predictor = Predictor(ts_model_path=self.model_folder + '/model.ts', output_classes_file=self.model_folder + '/classes.txt')
        print('Initialized ML Predictor')
        
    def stop_server(self, *args):
        """Stops the server
        """
        self.client.send_message("/terminated", "bang") 
        self.server.shutdown() 
        self.server.socket.close()
        
    def run(self):
        """run the server
        """
        self.server.serve_forever()
        print("Serving on {}".format(self.server.server_address))
        
    def init_bindings(self, osc_attributes=[]):
        """Here we define every OSC callbacks

        Args:
            osc_attributes (list, optional): _description_. Defaults to [].
        """
        self.dispatcher.map("bang", self.print_callback)
        self.dispatcher.map("/model/predict", self.predict_callback)

    def print_callback(self, addr, args):
        print("bang yoo")
        
    def send(self, address, content):
        """send message

        Args:
            address (_type_): _description_
            content (_type_): _description_
        """
        self.client.send_message(address=address, value=content)


    def predict_callback(self, addr, args=[]):
        """Calls predict() method and send resuts to Max. Keeps only top 5 tags.

        Args:
            addr (str): Client address to send to
        """
        print('Got predict signal')
        result_dict = self.predictor.predict()
        # Keep only top 5
        result_dict = {k: v for k, v in sorted(result_dict.items(), key=lambda item: item[1], reverse=True)[:5]}
        i=1
        for k, v in result_dict.items():
            self.send(address=f"/tag/{i}", content=f"{k}:{v}")
            i+=1
        self.send(address="/info", content="prediction done")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="127.0.0.1", help="The ip to listen on")
    parser.add_argument("--in_port", type=int, default=8001, help="The port to listen on")
    parser.add_argument("--out_port", type=int, default=8000, help="The port to send to")
    parser.add_argument("--model_folder", type=str, default='models/shortres_mtg', help="Path to model folder")

    args = parser.parse_args()
    
    # Initialize OSC Client/Server
    osc_server = OSCServer(in_port=args.in_port, out_port=args.out_port, ip=args.ip, model_folder=args.model_folder)
    osc_server.send("/info", "~~~ Hello from OSC server :) ~~~ ")
    osc_server.run()
    print("Server is running")
