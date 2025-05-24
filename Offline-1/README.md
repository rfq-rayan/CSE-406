# CSE406 Offline 1: AES and ECC-DH Cryptosystem

This project implements a cryptosystem using AES for symmetric encryption and Elliptic Curve Diffie-Hellman (ECC-DH) for secure key exchange between a client and a server.

## Prerequisites

*   Python 3.x
*   `pip` (Python package installer)

## Installation

1.  Clone or download this repository.
2.  Navigate to the project directory in your terminal.
3.  Install the required `sympy` library:
    ```bash
    pip install -r requirements.txt
    ```

## File Structure

*   [`_2005062_aes_implementation.py`](_2005062_aes_implementation.py): Standalone implementation of AES encryption/decryption with CBC and CTR modes. Supports text and file operations.
*   [`_2005062_ecc_dh.py`](_2005062_ecc_dh.py): Standalone implementation of Elliptic Curve Diffie-Hellman key exchange.
*   [`2005062_server.py`](2005062_server.py): Server-side implementation using TCP sockets. Receives ECDH parameters, generates its keys, computes the shared secret, derives the AES key, and decrypts messages from the client.
*   [`2005062_client.py`](2005062_client.py): Client-side implementation using TCP sockets. Initiates ECDH, sends parameters, computes the shared secret, derives the AES key, and sends an encrypted message to the server.
*   [`requirements.txt`](requirements.txt): Lists the Python package dependencies.
*   [`bitvector-demo.py`](bitvector-demo.py): Demonstration script for using the BitVector library (provided for reference).
*   `*.pdf`, `*.txt`: Assignment specification documents.

## How to Run

### 1. Standalone AES Implementation

The [`_2005062_aes_implementation.py`](_2005062_aes_implementation.py) script can be run independently for AES operations.

**Usage:**

```bash
python _2005062_aes_implementation.py [options]
```

**Options:**

*   `-k <key>` or `--key <key>`: Specify the encryption key (default: "Thats my Kung Fu").
*   `-p <plaintext>` or `--plaintext <plaintext>`: Encrypt the given text string.
*   `-e <input_file>` or `--encrypt <input_file>`: Encrypt the specified file.
*   `-d <input_file>` or `--decrypt <input_file>`: Decrypt the specified file.
*   `-o <output_file>` or `--output <output_file>`: Specify the output file path for encryption/decryption. Defaults are `<input_file>.enc` for encryption and `<input_file>.dec` for decryption.
*   `-m <mode>` or `--mode <mode>`: Choose encryption mode (`cbc` or `ctr`, default: `cbc`).
*   `-v` or `--verbose`: Enable detailed step-by-step output during encryption/decryption.

**Examples:**

*   Encrypt text using CBC mode:
    ```bash
    python _2005062_aes_implementation.py -p "My secret message" -k "mysecretpassword" -m cbc
    ```
*   Encrypt a file using CTR mode and verbose output:
    ```bash
    python _2005062_aes_implementation.py -e mydocument.txt -o mydocument.enc -k "anotherkey123456" -m ctr -v
    ```
*   Decrypt a file using CBC mode:
    ```bash
    python _2005062_aes_implementation.py -d mydocument.enc -o mydocument.dec -k "anotherkey123456" -m cbc
    ```

### 2. Standalone ECC-DH Implementation

The [`_2005062_ecc_dh.py`](_2005062_ecc_dh.py ) script demonstrates the ECC-DH key exchange and measures performance.

**Usage:**

```bash
python _2005062_ecc_dh.py [options]
```

**Options:**

*   `-v` or `--verbose`: Enable detailed step-by-step output during key generation and exchange.

**Example:**

*   Run the performance test (currently set for 128-bit keys):
    ```bash
    python _2005062_ecc_dh.py
    ```

### 3. Client-Server Cryptosystem

This demonstrates the full system with key exchange and encrypted communication.

**Steps:**

1.  **Start the Server:**
    Open a terminal and run the server script. You can specify the mode (`cbc` or `ctr`).
    ```bash
    # Using default CBC mode
    python 2005062_server.py

    # Using CTR mode
    python 2005062_server.py -m ctr
    ```
    The server will start listening for connections.

2.  **Start the Client:**
    Open a *second* terminal and run the client script. Ensure the mode matches the server's mode.
    ```bash
    # Using default CBC mode (if server is using CBC)
    python 2005062_client.py

    # Using CTR mode (if server is using CTR)
    python 2005062_client.py -m ctr
    ```
    The client will connect to the server, perform the ECC-DH key exchange, derive the AES key, encrypt a predefined message ("Hello, this is a secret message!"), and send it to the server. The server will then decrypt and print the message. Both client and server will print detailed steps of the process.