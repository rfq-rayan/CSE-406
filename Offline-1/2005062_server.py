import socket
import pickle
from _2005062_ecc_dh import ECDH
import time
from _2005062_aes_implementation import AES
import argparse

def derive_aes_key(shared_secret):
    """Derive a 16-byte AES key from the shared secret"""
    
    x_coord = str(shared_secret[0]) # Convert to string 
    return x_coord[:16].ljust(16, '0') # 16 byte

def start_server(host='localhost', port=12345, mode='cbc'):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    print(f"[SERVER] Listening on {host}:{port}")

    client_socket, addr = server_socket.accept()
    print(f"[SERVER] Connected to client at {addr}")
    time.sleep(3)

    try:
        # ecdh
        print("[SERVER] Receiving ECDH parameters...")
        params = pickle.loads(client_socket.recv(4096))
        P, a, b, G, A_public = params
        print(f"[SERVER] Received parameters: P={P}, a={a}, b={b}, G={G}, A_public={A_public}")
        time.sleep(3)

        # ecdh bob
        bob = ECDH(verbose=False)
        bob.P = P
        bob.a = a
        bob.b = b
        bob.G = G
        time.sleep(3)

        # bob public key
        print("[SERVER] Generating Bob's keys...")
        B_public = bob.generate_keys()
        print(f"[SERVER] Bob's public key: {B_public}")
        time.sleep(3)

        # send bob public key
        print("[SERVER] Sending Bob's public key to Alice...")
        client_socket.send(pickle.dumps(B_public))
        time.sleep(3)

        # shared secret
        print("[SERVER] Computing shared secret...")
        shared_secret = bob.compute_shared_secret(A_public)
        print(f"[SERVER] Shared secret: {shared_secret}")
        time.sleep(3)

        # aes key
        aes_key = derive_aes_key(shared_secret)
        print(f"[SERVER] Derived AES key: {aes_key}")
        time.sleep(3)

        #ready signal
        print("[SERVER] Sending ready signal...")
        client_socket.send(b"READY")
        time.sleep(3)

        # rcv msg
        print("[SERVER] Waiting for encrypted message...")
        encrypted_data = client_socket.recv(4096)
        print(f"[SERVER] Received encrypted data: {encrypted_data.hex()}")
        time.sleep(3)
        
        #aes decryption
        aes = AES(aes_key, verbose=False)
        if mode == 'ctr':
            # Extract nonce (12 bytes) and ciphertext
            nonce = encrypted_data[:12]
            ciphertext = encrypted_data[12:]
            decrypted_message = aes.decrypt_ctr(nonce, ciphertext)
        else:  # cbc mode
            # Extract IV (16 bytes) and ciphertext
            iv = encrypted_data[:16]
            ciphertext = encrypted_data[16:]
            decrypted_message = aes.decrypt(iv, ciphertext)
            
        print(f"[SERVER] Decrypted message: {decrypted_message}")
        time.sleep(3)

    except Exception as e:
        print(f"[SERVER] Error: {e}")
    finally:
        client_socket.close()
        server_socket.close()
        print("[SERVER] Connection closed")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AES Server')
    parser.add_argument('-m', '--mode', choices=['cbc', 'ctr'], default='cbc',
                      help='Encryption mode (default: cbc)')
    args = parser.parse_args()
    start_server(mode=args.mode) 