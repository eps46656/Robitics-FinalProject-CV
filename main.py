import socket
from Server import Server
# from Estimator import Estimator

class GLOBAL_CLASS: pass
GLOBAL = GLOBAL_CLASS()

def SessionFuncA(_, conn, addr):
    print(f"start serving connection from {addr}")

    conn.settimeout(2)

    while GLOBAL.server.is_active():
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
    GLOBAL.addr = socket.gethostname()
    GLOBAL.port = 8910

    GLOBAL.server = Server(GLOBAL.addr, GLOBAL.port, SessionFuncA)
    # GLOBAL.es = Estimator()

    GLOBAL.server.Start()

    try:
        input("")
    except:
        pass

    GLOBAL.server.Stop()

if __name__ == "__main__":
    main()
