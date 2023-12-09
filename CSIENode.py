import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")

import config
import cv2 as cv
import socket

from Server import *
from SocketBuffer import *

def SessionFunc(server, conn, addr, end_callback):
    sb_byte = SocketBufferByte(conn)

    while server.is_active():
        try:
            x = int.from_bytes(sb_byte.Recv(8, time.time() + 512), "little")
        except TimeoutException:
            continue
        except Exception as e:
            print(f"exception when receiving image: {e}")
            break

        print(f"{addr}: {x}")

    end_callback()

def main2():
    host = socket.gethostname()
    port = config.PORT

    server = Server(host, port, SessionFunc)

    server.Start()

    try:
        input()
    except:
        pass

    server.Stop()

def WaitToConnect(s, name):
    s.settimeout(10)

    try:
        while True:
            try:
                print(f"start to accept {name} connection")
                conn, addr = s.accept()
            except socket.timeout:
                print(f"timeout, retrying")
                continue
            except Exception as e:
                print(f"exception message: {e}")
                return
            break
    except Exception as e:
        print(f"exception message: {e}")
        return

    print(f"connect successfully from {name} at {addr}")

    return conn, addr

def main():
    host = ""
    port = config.PORT

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host, port))
    s.listen(8)

    print(f"csie node is OK")

    pe_conn, pe_addr = WaitToConnect(s, "pe node")

    pe_sb_byte = SocketBufferByte(pe_conn)
    pe_sb_arr = SocketBufferArr(pe_sb_byte)
    pe_sb_img = SocketBufferImage(pe_sb_byte)

    src_conn = None
    src_addr = None

    while True:
        if src_conn is None:
            src_conn, src_addr = WaitToConnect(s, "src node")

            src_sb_byte = SocketBufferByte(src_conn)
            src_sb_arr = SocketBufferArr(src_sb_byte)
            src_sb_img = SocketBufferImage(src_sb_byte)

        try:
            print(f"receiving img from src node")
            img = src_sb_img.Recv(None)
        except DisconnectionException:
            print(f"disconnect from src node")
            src_conn = None
            continue
        except Exception as e:
            print(f"exception message: {e}")
            return

        print(f"received img from src node")
        print(f"sending img to pe node")

        pe_sb_img.Send(img)

        try:
            print(f"receiving poses and scores from pe node")
            poses = pe_sb_arr.Recv(None)
            scores = pe_sb_arr.Recv(None)
        except Exception as e:
            print(f"exception message: {e}")
            return

        print(f"received poses and scores from pe node")
        print(f"sending poses and scores to pe node")

        try:
            src_sb_arr.Send(poses)
            src_sb_arr.Send(scores)
        except DisconnectionException:
            print(f"disconnect from src node")
            src_conn = None
            continue
        except Exception as e:
            print(f"exception message: {e}")

    s.close()

if __name__ == "__main__":
    main()
