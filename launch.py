import argparse
import time
import sys
import os
from multiprocessing import Process

from rcrs_core.connection.componentLauncher import ComponentLauncher
from rcrs_core.constants.constants import DEFAULT_KERNEL_PORT_NUMBER, DEFAULT_KERNEL_HOST_NAME

from rcrs_ddcop.core.policeForceAgent import PoliceForceAgent
from rcrs_ddcop.core.ambulanceTeamAgent import AmbulanceTeamAgent
from rcrs_ddcop.core.fireBrigadeAgent import FireBrigadeAgent
from rcrs_ddcop.core.fireStationAgent import FireStationAgent
from rcrs_ddcop.core.policeOfficeAgent import PoliceOfficeAgent
from rcrs_ddcop.core.ambulanceCenterAgent import AmbulanceCenterAgent


class Launcher:
    def __init__(self, ):
        pass        

    def launch(self, agent, _request_id):
        self.component_launcher.connect(agent, _request_id)

    def run(self, kwargs):
        processes = []
        agents = {}
        self.component_launcher = ComponentLauncher(kwargs['port'], kwargs['host'])
        agents['FireBrigadeAgent'] = kwargs['fb'] if kwargs['fb'] >= 0 else 100
        agents['FireStationAgent'] = kwargs['fs'] if kwargs['fs'] >= 0 else 100
        agents['PoliceForceAgent'] = kwargs['pf'] if kwargs['pf'] >= 0 else 100
        agents['PoliceOfficeAgent'] = kwargs['po'] if kwargs['po'] >= 0 else 100
        agents['AmbulanceTeamAgent'] = kwargs['at'] if kwargs['at'] >= 0 else 100
        agents['AmbulanceCenterAgent'] = kwargs['ac'] if kwargs['ac'] >= 0 else 100
        precompute = kwargs['precompute']

        for agn, num in agents.items():
            for _ in range(num):
                request_id = self.component_launcher.generate_request_ID()
                process = Process(target=self.launch, args=(eval(agn)(precompute), request_id))
                process.start()
                processes.append(process)
                time.sleep(1/100)
        
        for p in processes:
            p.join()


def main(sys_args):
    if not os.path.exists('logs'):
        os.makedirs('logs')

    filelist = [f for f in os.listdir('logs') if f.endswith(".log")]
    for f in filelist:
        os.remove(os.path.join('logs', f))

    print("start launcher...")
    l = Launcher()
    l.run(sys_args.__dict__)
    
    while True:
        try:
            time.sleep(2)
        except KeyboardInterrupt:
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--host', type=str, default=DEFAULT_KERNEL_HOST_NAME, help='RCRS server host address/IP')
    parser.add_argument('--port', type=int, default=DEFAULT_KERNEL_PORT_NUMBER, help='RCRS sever port number')
    parser.add_argument('-fb', type=int, default=0, help='Number of fire agents')
    parser.add_argument('-fs', type=int, default=0, help='Number of fire stations')
    parser.add_argument('-pf', type=int, default=0, help='Number of police agents')
    parser.add_argument('-po', type=int, default=0, help='Number of police offices')
    parser.add_argument('-at', type=int, default=0, help='Number of ambulance agents')
    parser.add_argument('-ac', type=int, default=0, help='Number of ambulance offices')
    parser.add_argument('--precompute', type=bool, default=False, help='Precomputation flag. Defaults to false')

    args = parser.parse_args()

    main(args)

    
