import socket


def main():
    s = socket.socket()
    host = socket.gethostname()
    port = 8910
    s.connect((host, port))

    while True:
        x = int(input("x: "))
        print(f"out x: {x}")
        s.send(x.to_bytes(8, "little"))

        if x == 0:
            break

    s.close()

if __name__ == "__main__":
    main()
