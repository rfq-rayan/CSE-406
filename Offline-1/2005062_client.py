import socket
import pickle
from _2005062_ecc_dh import ECDH
import time
from _2005062_aes_implementation import AES
import argparse

def derive_aes_key(shared_secret):
    x_coord = str(shared_secret[0]) # Convert to string 
    return x_coord[:16].ljust(16, '0') # 16 byte

def start_client(host='localhost', port=12345, mode='cbc'):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        print(f"[CLIENT] Connecting to server at {host}:{port}...")
        client_socket.connect((host, port))
        print("[CLIENT] Connected to server")
        time.sleep(3)

        #ECDH for Alice
        print("[CLIENT] Initializing ECDH...")
        alice = ECDH(verbose=False)
        alice.setup()
        time.sleep(3)
        
        print("[CLIENT] Generating Alice's keys...")
        A_public = alice.generate_keys() # alice public key
        print(f"[CLIENT] Alice's public key: {A_public}")
        time.sleep(3)

        #ECDH  to Bob
        print("[CLIENT] Sending ECDH parameters to Bob...")
        params = (alice.P, alice.a, alice.b, alice.G, A_public)
        client_socket.send(pickle.dumps(params))
        time.sleep(3)

        # receive bob public key
        print("[CLIENT] Waiting for Bob's public key...")
        B_public = pickle.loads(client_socket.recv(4096))
        print(f"[CLIENT] Received Bob's public key: {B_public}")
        time.sleep(3)

        #shared secret
        print("[CLIENT] Computing shared secret...")
        shared_secret = alice.compute_shared_secret(B_public)
        print(f"[CLIENT] Shared secret: {shared_secret}")
        time.sleep(3)

        # aes key find
        aes_key = derive_aes_key(shared_secret)
        print(f"[CLIENT] Derived AES key: {aes_key}")
        time.sleep(3)

        print("[CLIENT] Waiting for ready signal...")
        ready_signal = client_socket.recv(1024)
        if ready_signal != b"READY":
            raise Exception("Did not receive ready signal from server")
        time.sleep(3)

        message = "Hello, this is a secret message!"
        print(f"[CLIENT] Original message: {message}")
        time.sleep(3)

        #aes encryption
        aes = AES(aes_key, verbose=False)
        if mode == 'ctr':
            nonce, encrypted_message = aes.encrypt_ctr(message)
            print(f"[CLIENT] Encrypted message (hex): { ' '.join(f'{b:02X}' for b in nonce + encrypted_message)}")
            print("[CLIENT] Sending encrypted message...")
            client_socket.send(nonce + encrypted_message)
        else:  # cbc mode
            iv, encrypted_message = aes.encrypt(message)
            print(f"[CLIENT] Encrypted message (hex): { ' '.join(f'{b:02X}' for b in iv + encrypted_message)}")
            print("[CLIENT] Sending encrypted message...")
            client_socket.send(iv + encrypted_message)
        print("[CLIENT] Message sent successfully")
        time.sleep(3)

    except Exception as e:
        print(f"[CLIENT] Error: {e}")
    finally:
        client_socket.close()
        print("[CLIENT] Connection closed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AES Client')
    parser.add_argument('-m', '--mode', choices=['cbc', 'ctr'], default='cbc',
                      help='Encryption mode (default: cbc)')
    args = parser.parse_args()
    start_client(mode=args.mode) 