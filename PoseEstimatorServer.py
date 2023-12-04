import config
import socket
from Server import Server
# from Estimator import Estimator

class PoseEstimatorServer:
    def __init__(self, addr, port):
        self.server = Server(addr, port, self.SessionFuncA)
        # GLOBAL.es = Estimator()

    def Start(self):
        self.server.Start()

    def Stop(self):
        self.server.Stop()

    def SessionFuncA(self, _, conn, addr):
        print(f"start serving connection from {addr}")

        conn.settimeout(2)

        while self.server.is_active():
            try:
                x = int.from_bytes(conn.recv(1024), "little")
                print(f"{addr}: received {x}")

                if x == 0:
                    break
            except socket.timeout:
                continue
            except Exception as e:
                print(f"exception message: {e}")
                break

        print(f"stop serving connection from {addr}")

def main():
    addr = socket.gethostname()
    port = config.PORT

    pose_es = PoseEstimatorServer(addr, port)

    pose_es.Start()

    try:
        input("")
    except:
        pass

    pose_es.Stop()

if __name__ == "__main__":
    main()
