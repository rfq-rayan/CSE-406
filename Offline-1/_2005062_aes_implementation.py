import time
from typing import List, Tuple, Union
import os
import argparse
import mimetypes
from concurrent.futures import ThreadPoolExecutor

# constants 
SBOX = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76,
    0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0,
    0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15,
    0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75,
    0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84,
    0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf,
    0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8,
    0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2,
    0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73,
    0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb,
    0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79,
    0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08,
    0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a,
    0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e,
    0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16
]

INV_SBOX = [
    0x52, 0x09, 0x6a, 0xd5, 0x30, 0x36, 0xa5, 0x38, 0xbf, 0x40, 0xa3, 0x9e, 0x81, 0xf3, 0xd7, 0xfb,
    0x7c, 0xe3, 0x39, 0x82, 0x9b, 0x2f, 0xff, 0x87, 0x34, 0x8e, 0x43, 0x44, 0xc4, 0xde, 0xe9, 0xcb,
    0x54, 0x7b, 0x94, 0x32, 0xa6, 0xc2, 0x23, 0x3d, 0xee, 0x4c, 0x95, 0x0b, 0x42, 0xfa, 0xc3, 0x4e,
    0x08, 0x2e, 0xa1, 0x66, 0x28, 0xd9, 0x24, 0xb2, 0x76, 0x5b, 0xa2, 0x49, 0x6d, 0x8b, 0xd1, 0x25,
    0x72, 0xf8, 0xf6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xd4, 0xa4, 0x5c, 0xcc, 0x5d, 0x65, 0xb6, 0x92,
    0x6c, 0x70, 0x48, 0x50, 0xfd, 0xed, 0xb9, 0xda, 0x5e, 0x15, 0x46, 0x57, 0xa7, 0x8d, 0x9d, 0x84,
    0x90, 0xd8, 0xab, 0x00, 0x8c, 0xbc, 0xd3, 0x0a, 0xf7, 0xe4, 0x58, 0x05, 0xb8, 0xb3, 0x45, 0x06,
    0xd0, 0x2c, 0x1e, 0x8f, 0xca, 0x3f, 0x0f, 0x02, 0xc1, 0xaf, 0xbd, 0x03, 0x01, 0x13, 0x8a, 0x6b,
    0x3a, 0x91, 0x11, 0x41, 0x4f, 0x67, 0xdc, 0xea, 0x97, 0xf2, 0xcf, 0xce, 0xf0, 0xb4, 0xe6, 0x73,
    0x96, 0xac, 0x74, 0x22, 0xe7, 0xad, 0x35, 0x85, 0xe2, 0xf9, 0x37, 0xe8, 0x1c, 0x75, 0xdf, 0x6e,
    0x47, 0xf1, 0x1a, 0x71, 0x1d, 0x29, 0xc5, 0x89, 0x6f, 0xb7, 0x62, 0x0e, 0xaa, 0x18, 0xbe, 0x1b,
    0xfc, 0x56, 0x3e, 0x4b, 0xc6, 0xd2, 0x79, 0x20, 0x9a, 0xdb, 0xc0, 0xfe, 0x78, 0xcd, 0x5a, 0xf4,
    0x1f, 0xdd, 0xa8, 0x33, 0x88, 0x07, 0xc7, 0x31, 0xb1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xec, 0x5f,
    0x60, 0x51, 0x7f, 0xa9, 0x19, 0xb5, 0x4a, 0x0d, 0x2d, 0xe5, 0x7a, 0x9f, 0x93, 0xc9, 0x9c, 0xef,
    0xa0, 0xe0, 0x3b, 0x4d, 0xae, 0x2a, 0xf5, 0xb0, 0xc8, 0xeb, 0xbb, 0x3c, 0x83, 0x53, 0x99, 0x61,
    0x17, 0x2b, 0x04, 0x7e, 0xba, 0x77, 0xd6, 0x26, 0xe1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0c, 0x7d
]

RCON = [0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36]

class AES:
    def __init__(self, key: str, verbose: bool = False):
        """
        initialize aes cipher
        Args:
            key: Encryption key (should be 16, 24, or 32 characters)
            verbose: Whether to print debug information
        """
        self.verbose = verbose
        # Convert key to bytes
        key_bytes = key.encode('utf-8')
        
        key_length = len(key_bytes)
        
        # Determine key size and number of rounds
        if key_length <= 16:
            self.Nk = 4  # Number of 32-bit words in key
            self.Nr = 10  # Number of rounds
            # Pad key to exactly 16 bytes
            self.key = key_bytes.ljust(16, b'\0')
        elif key_length <= 24:
            self.Nk = 6
            self.Nr = 12
            # Pad key to exactly 24 bytes
            self.key = key_bytes.ljust(24, b'\0')
        else:
            self.Nk = 8
            self.Nr = 14
            # Truncate or pad key to exactly 32 bytes
            self.key = key_bytes[:32].ljust(32, b'\0')
        
        self.key_schedule = self._key_expansion()
        
    def _rot_word(self, word: bytes) -> bytes:
        """
        Rotate word left by one byte
        """
        return word[1:] + word[:1]

    def _sub_word(self, word: bytes) -> bytes:
        """
        Substitute each byte in word using S-box
        """
        return bytes(SBOX[b] for b in word)

    def _key_expansion(self) -> List[bytes]:
        """
        Expand the key into round keys
        Returns:
            List of round keys
        """
        Nb = 4  # Number of columns in state (4 for AES)
        
        # Convert key to list of 32-bit words
        key_words = [self.key[i:i+4] for i in range(0, len(self.key), 4)]
        
        # Generate expanded key
        for i in range(self.Nk, Nb * (self.Nr + 1)):
            temp = bytearray(key_words[i-1])
            
            if i % self.Nk == 0:
                # Rotate word left by one byte
                temp = temp[1:] + temp[:1]
                # Apply S-box
                temp = bytes(SBOX[b] for b in temp)
                # XOR with round constant
                temp = bytes([temp[0] ^ RCON[i // self.Nk]] + list(temp[1:]))
            elif self.Nk > 6 and i % self.Nk == 4:
                # Additional transformation for 256-bit keys
                temp = bytes(SBOX[b] for b in temp)
            
            # XOR with word self.Nk positions earlier
            word = bytes(a ^ b for a, b in zip(key_words[i-self.Nk], temp))
            key_words.append(word)
        
        # Convert words to round keys
        round_keys = []
        for i in range(self.Nr + 1):
            round_key = b''
            for j in range(Nb):
                idx = i*Nb + j
                if idx < len(key_words):
                    round_key += key_words[idx]
            round_keys.append(round_key)
        # print the hex of the round keys in a table
        # print("Round keys:")
        # for i in range(self.Nr + 1):
        #     print(f"Round {i}: {round_keys[i].hex()}")
        # print("--------------------------------")
        
        return round_keys

    def _sub_bytes(self, state: List[List[int]], inverse: bool = False) -> List[List[int]]:
        """
        SubBytes transformation
        Args:
            state: 4x4 state matrix
            inverse: Whether to perform inverse transformation
        Returns:
            Transformed state matrix
        """
        sbox = INV_SBOX if inverse else SBOX
        return [[sbox[byte] for byte in row] for row in state]

    def _shift_rows(self, state: List[List[int]], inverse: bool = False) -> List[List[int]]:
        """
        ShiftRows transformation
        Args:
            state: 4x4 state matrix
            inverse: Whether to perform inverse transformation
        Returns:
            Transformed state matrix
        """
        if not inverse:
            # Shift left
            return [
                state[0],
                state[1][1:] + state[1][:1],
                state[2][2:] + state[2][:2],
                state[3][3:] + state[3][:3]
            ]
        else:
            # Shift right
            return [
                state[0],
                state[1][-1:] + state[1][:-1],
                state[2][-2:] + state[2][:-2],
                state[3][-3:] + state[3][:-3]
            ]

    def _mix_columns(self, state: List[List[int]], inverse: bool = False) -> List[List[int]]:
        """
        MixColumns transformation
        Args:
            state: 4x4 state matrix
            inverse: Whether to perform inverse transformation
        Returns:
            Transformed state matrix
        """
        def mul2(x):
            return ((x << 1) ^ (0x1B if x & 0x80 else 0)) & 0xFF

        def mul3(x):
            return mul2(x) ^ x

        def mul9(x):
            return mul2(mul2(mul2(x))) ^ x

        def mul11(x):
            return mul2(mul2(mul2(x))) ^ mul2(x) ^ x

        def mul13(x):
            return mul2(mul2(mul2(x))) ^ mul2(mul2(x)) ^ x

        def mul14(x):
            return mul2(mul2(mul2(x))) ^ mul2(mul2(x)) ^ mul2(x)

        if not inverse:
            # Forward MixColumns
            new_state = [[0] * 4 for _ in range(4)]
            for i in range(4):
                new_state[0][i] = mul2(state[0][i]) ^ mul3(state[1][i]) ^ state[2][i] ^ state[3][i]
                new_state[1][i] = state[0][i] ^ mul2(state[1][i]) ^ mul3(state[2][i]) ^ state[3][i]
                new_state[2][i] = state[0][i] ^ state[1][i] ^ mul2(state[2][i]) ^ mul3(state[3][i])
                new_state[3][i] = mul3(state[0][i]) ^ state[1][i] ^ state[2][i] ^ mul2(state[3][i])
        else:
            # Inverse MixColumns
            new_state = [[0] * 4 for _ in range(4)]
            for i in range(4):
                new_state[0][i] = mul14(state[0][i]) ^ mul11(state[1][i]) ^ mul13(state[2][i]) ^ mul9(state[3][i])
                new_state[1][i] = mul9(state[0][i]) ^ mul14(state[1][i]) ^ mul11(state[2][i]) ^ mul13(state[3][i])
                new_state[2][i] = mul13(state[0][i]) ^ mul9(state[1][i]) ^ mul14(state[2][i]) ^ mul11(state[3][i])
                new_state[3][i] = mul11(state[0][i]) ^ mul13(state[1][i]) ^ mul9(state[2][i]) ^ mul14(state[3][i])
        
        return new_state

    def _add_round_key(self, state: List[List[int]], round_key: bytes) -> List[List[int]]:
        """
        AddRoundKey transformation
        Args:
            state: 4x4 state matrix
            round_key: Round key to XOR with
        Returns:
            Transformed state matrix
        """
        # Convert round key to 4x4 matrix
        key_matrix = []
        for i in range(4):
            row = []
            for j in range(4):
                if i + 4*j < len(round_key):
                    row.append(round_key[i + 4*j])
                else:
                    row.append(0)  # Pad with zeros if round key is too short
            key_matrix.append(row)
        
        # XOR state with round key
        return [[state[i][j] ^ key_matrix[i][j] for j in range(4)] for i in range(4)]

    def _bytes_to_state(self, data: bytes) -> List[List[int]]:
        """
        Convert 16 bytes to 4x4 state matrix
        """
        state = [[0 for _ in range(4)] for _ in range(4)]
        for i in range(4):
            for j in range(4):
                if i + 4*j < len(data):
                    state[i][j] = data[i + 4*j]
        return state

    def _state_to_bytes(self, state: List[List[int]]) -> bytes:
        """
        Convert 4x4 state matrix to 16 bytes
        """
        return bytes(state[i][j] for j in range(4) for i in range(4))

    def _pad(self, data: bytes) -> bytes:
        """
        PKCS#7 padding
        """
        padding_length = 16 - (len(data) % 16)
        padding = bytes([padding_length] * padding_length)
        return data + padding

    def _unpad(self, data: bytes) -> bytes:
        """
        Remove PKCS#7 padding
        """
        if not data:
            return b''
            
        padding_length = data[-1]
        
        # Validate padding
        if padding_length > 16 or padding_length == 0:
            return data  # Invalid padding, return as is
            
        # Check that all padding bytes are correct
        for i in range(1, padding_length + 1):
            if i <= len(data) and data[-i] != padding_length:
                return data  # Invalid padding, return as is
                
        return data[:-padding_length]

    def _encrypt_block(self, block: bytes) -> bytes:
        """
        Encrypt a single 16-byte block
        """
        # Ensure block is exactly 16 bytes
        if len(block) < 16:
            block = block + b'\0' * (16 - len(block))
        
        elif len(block) > 16:
            block = block[:16]
        state = self._bytes_to_state(block)
        if self.verbose:
            print(f"Block starting encryption  : {' '.join(f'{b:02x}' for b in block)}")
        
        # Initial round
        state = self._add_round_key(state, self.key_schedule[0])
        if self.verbose:
            print("After initial round:")
            for row in state:
                print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        
        # Main rounds
        for round_num in range(1, self.Nr):
            state = self._sub_bytes(state)
            if self.verbose:
                print(f"After sub bytes:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
            state = self._shift_rows(state)
            if self.verbose:
                print(f"After shift rows:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
            state = self._mix_columns(state)
            if self.verbose:
                print(f"After mix columns:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
            state = self._add_round_key(state, self.key_schedule[round_num])
            if self.verbose:
                print(f"After xor round {round_num}:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        
        # Final round
        state = self._sub_bytes(state)
        state = self._shift_rows(state)
        state = self._add_round_key(state, self.key_schedule[self.Nr])
        if self.verbose:
            print(f"After final round:")
            for row in state:
                print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        return self._state_to_bytes(state)

    def _decrypt_block(self, block: bytes) -> bytes:
        """
        Decrypt a single 16-byte block
        """
        # Ensure block is exactly 16 bytes
        if len(block) < 16:
            block = block + b'\0' * (16 - len(block))
        elif len(block) > 16:
            block = block[:16]
            
        state = self._bytes_to_state(block)
        if self.verbose:
            print(f"Block starting decryption: {' '.join(f'{b:02x}' for b in block)}")
        
        # Initial round
        state = self._add_round_key(state, self.key_schedule[self.Nr])
        if self.verbose:
            print("After initial round:")
            for row in state:
                print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        state = self._shift_rows(state, inverse=True)
        if self.verbose:
            print("After inverse shift rows:")
            for row in state:
                print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        state = self._sub_bytes(state, inverse=True)
        if self.verbose:
            print("After inverse sub bytes:")
            for row in state:
                print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        
        # Main rounds
        for round_num in range(self.Nr-1, 0, -1):
            state = self._add_round_key(state, self.key_schedule[round_num])
            if self.verbose:
                print(f"After xor round {round_num}:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
            state = self._mix_columns(state, inverse=True)
            if self.verbose:
                print("After inverse mix columns:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
            state = self._shift_rows(state, inverse=True)
            if self.verbose:
                print("After inverse shift rows:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
            state = self._sub_bytes(state, inverse=True)
            if self.verbose:
                print("After inverse sub bytes:")
                for row in state:
                    print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        
        # Final round
        state = self._add_round_key(state, self.key_schedule[0])
        if self.verbose:
            print("After final round:")
            for row in state:
                print(f"{row[0]:02x}  {row[1]:02x}  {row[2]:02x}  {row[3]:02x}")
        
        return self._state_to_bytes(state)

    def encrypt(self, plaintext: str) -> Tuple[bytes, bytes]:
        """
        Encrypt plaintext using AES-CBC
        Args:
            plaintext: Text to encrypt
        Returns:
            Tuple of (IV, ciphertext)
        """
        # Generate random IV
        iv = os.urandom(16)
        # iv = b'\x00' * 16
        # Convert plaintext to bytes and pad
        data = plaintext.encode('utf-8')
        
        print(f"Data: {' '.join(f'{b:02x}' for b in data)}")

        padded_data = self._pad(data)
        
        print(f"Padded data: {' '.join(f'{b:02x}' for b in padded_data)}")
        # Split into blocks
        blocks = [padded_data[i:i+16] for i in range(0, len(padded_data), 16)]
        # CBC mode encryption
        ciphertext = b''
        prev_block = iv
        
        for block in blocks:
            # XOR with previous block
            xored = bytes(a ^ b for a, b in zip(block, prev_block.ljust(len(block), b'\0')))
            if self.verbose:
                print(f"Xored: {' '.join(f'{b:02x}' for b in xored)}")
            # Encrypt
            encrypted = self._encrypt_block(xored)
            ciphertext += encrypted
            prev_block = encrypted
        # print the cipher text as iv + cipher text

        print(f"Cipher Text: {' '.join(f'{b:02x}' for b in iv + ciphertext)}")
        print(f"In ASCII: {ciphertext.decode('utf-8', errors='replace')}")
        print()

        return iv, ciphertext

    def decrypt(self, iv: bytes, ciphertext: bytes) -> str:
        """
        Decrypt ciphertext using AES-CBC
        Args:
            iv: Initialization vector
            ciphertext: Encrypted data
        Returns:
            Decrypted plaintext
        """
        if len(ciphertext) % 16 != 0:
            raise ValueError("Ciphertext length must be multiple of 16")
            
        # Split into blocks
        blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
        if self.verbose:
            print(f"Number of blocks to decrypt: {len(blocks)}")
        
        # CBC mode decryption
        plaintext = b''
        prev_block = iv
        
        for i, block in enumerate(blocks):
            if self.verbose:
                print(f"\nDecrypting block {i+1}:")
                print(f"Block: {' '.join(f'{b:02x}' for b in block)}")
                print(f"Previous block: {' '.join(f'{b:02x}' for b in prev_block)}")
            
            # Decrypt block
            decrypted = self._decrypt_block(block)
            if self.verbose:
                print(f"After block decryption: {' '.join(f'{b:02x}' for b in decrypted)}")
            
            # XOR with previous ciphertext block
            xored = bytes(a ^ b for a, b in zip(decrypted, prev_block.ljust(len(decrypted), b'\0')))
            if self.verbose:
                print(f"After XOR with previous block: {' '.join(f'{b:02x}' for b in xored)}")
            
            plaintext += xored
            prev_block = block
        print(f"Before unpadding: {' '.join(f'{b:02x}' for b in plaintext)}")
        print(f"In ASCII: {plaintext.decode('utf-8', errors='replace')}")
        # Remove padding and convert to string
        try:
            unpadded = self._unpad(plaintext)
            print(f"After unpadding: {' '.join(f'{b:02x}' for b in unpadded)}")
            print(f"In ASCII: {unpadded.decode('utf-8', errors='replace')}")
            if self.verbose:
                print(f"\nFinal plaintext (hex): {' '.join(f'{b:02x}' for b in unpadded)}")
            return unpadded.decode('utf-8')
        except UnicodeDecodeError as e:
            # Try to print the raw bytes for debugging
            print(f"Decoding error: {e}")
            print(f"Raw decrypted bytes: {' '.join(f'{b:02x}' for b in plaintext)}")
            return plaintext.decode('utf-8', errors='replace')

    def encrypt_file(self, input_file: str, output_file: str) -> None:
        """
        Encrypt a file using AES-CBC
        Args:
            input_file: Path to input file
            output_file: Path to output file
        """
        # Read input file
        with open(input_file, 'rb') as f:
            data = f.read()

        # Generate random IV
        iv = os.urandom(16)
        
        # Pad data
        padded_data = self._pad(data)
        
        # Split into blocks
        blocks = [padded_data[i:i+16] for i in range(0, len(padded_data), 16)]
        
        # CBC mode encryption
        ciphertext = b''
        prev_block = iv
        
        for block in blocks:
            # XOR with previous block
            xored = bytes(a ^ b for a, b in zip(block, prev_block.ljust(len(block), b'\0')))
            # Encrypt
            encrypted = self._encrypt_block(xored)
            ciphertext += encrypted
            prev_block = encrypted

        # Write IV and ciphertext to output file
        with open(output_file, 'wb') as f:
            f.write(iv + ciphertext)

    def decrypt_file(self, input_file: str, output_file: str) -> None:
        """
        Decrypt a file using AES-CBC
        Args:
            input_file: Path to input file
            output_file: Path to output file
        """
        # Read input file
        with open(input_file, 'rb') as f:
            data = f.read()

        # Extract IV and ciphertext
        iv = data[:16]
        ciphertext = data[16:]

        if len(ciphertext) % 16 != 0:
            raise ValueError("Ciphertext length must be multiple of 16")
            
        # Split into blocks
        blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
        
        # CBC mode decryption
        plaintext = b''
        prev_block = iv
        
        for block in blocks:
            # Decrypt block
            decrypted = self._decrypt_block(block)
            
            # XOR with previous ciphertext block
            xored = bytes(a ^ b for a, b in zip(decrypted, prev_block.ljust(len(decrypted), b'\0')))
            
            plaintext += xored
            prev_block = block

        # Remove padding
        unpadded = self._unpad(plaintext)

        # Write decrypted data to output file
        with open(output_file, 'wb') as f:
            f.write(unpadded)

    def _increment_counter(self, counter: bytes) -> bytes:
        """
        Increment a 16-byte counter
        """
        counter_int = int.from_bytes(counter, byteorder='big') # convert 16 byte counter to 128 bit integer, big endian ensures most significant byte is first
        counter_int = (counter_int + 1) % (1 << 128)  # after 2^128 back to 0
        return counter_int.to_bytes(16, byteorder='big') # back to 16 byte counter

    def encrypt_ctr(self, plaintext: str) -> Tuple[bytes, bytes]:
        """
        Encrypt plaintext using AES-CTR
        Args:
            plaintext: Text to encrypt
        Returns:
            Tuple of (nonce, ciphertext)
        """
        nonce = os.urandom(12)
        counter = b'\x00\x00\x00\x00'
        data = plaintext.encode('utf-8')
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]

        def process_block(args):
            block, idx = args
            current_counter = (int.from_bytes(counter, 'big') + idx) % (1 << 32)
            counter_block = nonce + current_counter.to_bytes(4, 'big')
            encrypted_counter = self._encrypt_block(counter_block)
            return bytes(a ^ b for a, b in zip(block, encrypted_counter[:len(block)]))

        with ThreadPoolExecutor() as executor:
            ciphertext_blocks = list(executor.map(process_block, [(block, idx) for idx, block in enumerate(blocks)]))

        ciphertext = b''.join(ciphertext_blocks)
        return nonce, ciphertext

    def decrypt_ctr(self, nonce: bytes, ciphertext: bytes) -> str:
        """
        Decrypt ciphertext using AES-CTR
        Args:
            nonce: Nonce used in encryption
            ciphertext: Encrypted data
        Returns:
            Decrypted plaintext
        """
        blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
        counter = b'\x00\x00\x00\x00'

        def process_block(args):
            block, idx = args
            current_counter = (int.from_bytes(counter, 'big') + idx) % (1 << 32)
            counter_block = nonce + current_counter.to_bytes(4, 'big')
            encrypted_counter = self._encrypt_block(counter_block)
            return bytes(a ^ b for a, b in zip(block, encrypted_counter[:len(block)]))

        with ThreadPoolExecutor() as executor:
            plaintext_blocks = list(executor.map(process_block, [(block, idx) for idx, block in enumerate(blocks)]))

        plaintext = b''.join(plaintext_blocks)
        return plaintext.decode('utf-8')

    def encrypt_file_ctr(self, input_file: str, output_file: str) -> None:
        """
        Encrypt a file using AES-CTR
        Args:
            input_file: Path to input file
            output_file: Path to output file
        """
        with open(input_file, 'rb') as f:
            data = f.read()
        nonce = os.urandom(12)
        counter = b'\x00\x00\x00\x00'
        blocks = [data[i:i+16] for i in range(0, len(data), 16)]

        def process_block(args):
            block, idx = args
            current_counter = (int.from_bytes(counter, 'big') + idx) % (1 << 32)
            counter_block = nonce + current_counter.to_bytes(4, 'big')
            encrypted_counter = self._encrypt_block(counter_block)
            return bytes(a ^ b for a, b in zip(block, encrypted_counter[:len(block)]))

        with ThreadPoolExecutor() as executor:
            ciphertext_blocks = list(executor.map(process_block, [(block, idx) for idx, block in enumerate(blocks)]))

        ciphertext = b''.join(ciphertext_blocks)
        with open(output_file, 'wb') as f:
            f.write(nonce + ciphertext)

    def decrypt_file_ctr(self, input_file: str, output_file: str) -> None:
        """
        Decrypt a file using AES-CTR
        Args:
            input_file: Path to input file
            output_file: Path to output file
        """
        # Read input file
        with open(input_file, 'rb') as f:
            data = f.read()

        # Extract nonce and ciphertext
        nonce = data[:12]
        ciphertext = data[12:]
        
        # Split into blocks
        blocks = [ciphertext[i:i+16] for i in range(0, len(ciphertext), 16)]
        counter = b'\x00\x00\x00\x00'

        def process_block(args):
            block, idx = args
            current_counter = (int.from_bytes(counter, 'big') + idx) % (1 << 32)
            counter_block = nonce + current_counter.to_bytes(4, 'big')
            encrypted_counter = self._encrypt_block(counter_block)
            return bytes(a ^ b for a, b in zip(block, encrypted_counter[:len(block)]))

        with ThreadPoolExecutor() as executor:
            plaintext_blocks = list(executor.map(process_block, [(block, idx) for idx, block in enumerate(blocks)]))

        plaintext = b''.join(plaintext_blocks)

        # Write decrypted data to output file
        with open(output_file, 'wb') as f:
            f.write(plaintext)

def main():
    """
    Test AES implementation with different key sizes and file operations
    """
    parser = argparse.ArgumentParser(description='AES Encryption/Decryption Tool')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('-k', '--key', default='Thats my Kung Fu', 
                      help='Encryption key (default: "Thats my Kung Fu")')
    parser.add_argument('-p', '--plaintext', help='Text to encrypt')
    parser.add_argument('-e', '--encrypt', help='File to encrypt')
    parser.add_argument('-d', '--decrypt', help='File to decrypt')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-m', '--mode', choices=['cbc', 'ctr'], default='cbc',
                      help='Encryption mode (default: cbc)')
    args = parser.parse_args()
    
    print("AES Encryption/Decryption Tool")
    print("==============================")
    
    # Create AES instance
    aes = AES(args.key, verbose=args.verbose)
    
    if args.encrypt:
        if not args.output:
            args.output = args.encrypt + '.enc'
        print(f"\nEncrypting file: {args.encrypt}")
        print(f"Output file: {args.output}")
        print(f"Mode: {args.mode.upper()}")
        start_time = time.perf_counter()
        if args.mode == 'ctr':
            aes.encrypt_file_ctr(args.encrypt, args.output)
        else:
            aes.encrypt_file(args.encrypt, args.output)
        encryption_time = time.perf_counter() - start_time
        print(f"Encryption Time: {encryption_time*1000000:.2f} microseconds")
        
    elif args.decrypt:
        if not args.output:
            args.output = args.decrypt + '.dec'
        print(f"\nDecrypting file: {args.decrypt}")
        print(f"Output file: {args.output}")
        print(f"Mode: {args.mode.upper()}")
        start_time = time.perf_counter()
        if args.mode == 'ctr':
            aes.decrypt_file_ctr(args.decrypt, args.output)
        else:
            aes.decrypt_file(args.decrypt, args.output)
        decryption_time = time.perf_counter() - start_time
        print(f"Decryption Time: {decryption_time*1000000:.2f} microseconds")
        
    elif args.plaintext:
        print(f"\nTesting with text input:")
        print(f"Key: {args.key}")
        print(f"In Hex: {' '.join(f'{b:02x}' for b in args.key.encode('utf-8'))}")
        print(f"Mode: {args.mode.upper()}")
        print()
        print(f"Plain Text: {args.plaintext}")
        
        start_time = time.perf_counter()
        if args.mode == 'ctr':
            nonce, ciphertext = aes.encrypt_ctr(args.plaintext)
        else:
            iv, ciphertext = aes.encrypt(args.plaintext)
        encryption_time = time.perf_counter() - start_time
        
        start_time = time.perf_counter()
        if args.mode == 'ctr':
            decrypted = aes.decrypt_ctr(nonce, ciphertext)
        else:
            decrypted = aes.decrypt(iv, ciphertext)
        decryption_time = time.perf_counter() - start_time
        
        print(f"Decrypted Text: {decrypted}")
        print(f"Encryption Time: {encryption_time*1000000:.2f} microseconds")
        print(f"Decryption Time: {decryption_time*1000000:.2f} microseconds")
    else:
        print("Please provide either text input (-p) or a file to encrypt (-e) or decrypt (-d)")

if __name__ == '__main__':
    main() 