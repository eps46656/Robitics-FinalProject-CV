import socket
import threading
from collections import deque

class Server:
    def __init__(self, addr, port, session_func):
        self.active = False

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((addr, port))
        self.socket.settimeout(2)
        self.socket.listen(1024)

        self.conns = deque()

        self.session_func = session_func

    def is_active(self):
        return self.active

    def Start(self):
        print(f"starting server")

        self.active = True

        self.accept_thread = threading.Thread(target=self.Accept)
        self.accept_thread.start()

        print(f"started")

    def Stop(self):
        print(f"stopping server")

        self.active = False

        self.accept_thread.join()
        self.Clean(len(self.conns), None)

        print(f"stopped")

    def Accept(self):
        while self.active:
            try:
                conn, addr = self.socket.accept()
            except socket.timeout:
                self.Clean(20, 1 / 1000) # try to join with 10ms timeout
                continue

            if not self.active:
                conn.close()
                break

            print(f"connection from {addr}")

            thread = threading.Thread(target=self.session_func,
                                      args=(self, conn, addr))

            thread.start()

            self.conns.append((conn, addr, thread))

    def Clean(self, max_query_num, timeout):
        max_query_num = min(max_query_num, len(self.conns))

        for _ in range(max_query_num):
            conn, addr, thread = self.conns[-1]

            thread.join(timeout)

            if thread.is_alive():
                self.conns.rotate()
            else:
                print(f"close connection from {addr}")
                conn.close()
                self.conns.pop()
