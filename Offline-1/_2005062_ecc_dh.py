import random
from sympy import randprime
import time
import argparse

def legendre_symbol(a, p):
    """Compute the Legendre symbol (a/p)"""
    ls = pow(a, (p - 1) // 2, p)
    return -1 if ls == p - 1 else ls

def tonelli_shanks(n, p):
    """Find square root of n modulo p using Tonelli-Shanks algorithm"""
    # Check if n is a quadratic residue
    if legendre_symbol(n, p) != 1:
        return None
    
    # Handle special cases
    if p == 2:
        return n
    if p % 4 == 3:
        return pow(n, (p + 1) // 4, p)
    
    # Find Q and S such that p-1 = Q*2^S
    Q = p - 1
    S = 0
    while Q % 2 == 0:
        Q //= 2
        S += 1
    
    # Find a quadratic non-residue z
    z = 2
    while legendre_symbol(z, p) != -1:
        z += 1
    
    # Initialize variables
    m = S
    c = pow(z, Q, p)
    t = pow(n, Q, p)
    r = pow(n, (Q + 1) // 2, p)
    
    # Main loop
    while t != 1:
        # Find the least i such that t^(2^i) = 1
        i = 0
        temp = t
        while temp != 1 and i < m:
            temp = pow(temp, 2, p)
            i += 1
        
        if i == 0:
            return r
        
        # Update variables
        b = pow(c, 1 << (m - i - 1), p)
        m = i
        c = pow(b, 2, p)
        t = (t * c) % p
        r = (r * b) % p
    
    return r

class ECDH:
    def __init__(self, key_size=128, verbose=False):
        self.key_size = key_size
        self.P = None
        self.a = None
        self.b = None
        self.G = None
        self.private_key = None
        self.public_key = None
        self.verbose = verbose

    def generate_prime(self): # prime size 128

        min_p = 2**(self.key_size-1)
        max_p = 2**self.key_size - 1
        P = randprime(min_p, max_p)
        if self.verbose:
            print(f"[DEBUG] Generated prime P: {P}")
        return P

    def generate_curve_parameters(self):
        attempts = 0
        while True:
            self.a = random.randrange(0, self.P)
            self.b = random.randrange(0, self.P)
            attempts += 1
            if (4 * pow(self.a, 3, self.P) + 27 * pow(self.b, 2, self.P)) % self.P != 0:
                if self.verbose:
                    # print(f"[DEBUG] Found valid curve parameters after {attempts} attempts:")
                    print(f"[DEBUG] a = {self.a}")
                    print(f"[DEBUG] b = {self.b}")
                break

    def find_point_on_curve(self):
        attempts = 0
        while True:
            x = random.randrange(0, self.P)
            # y^2 = x^3 + ax + b (mod P)
            rhs = (pow(x, 3, self.P) + self.a * x + self.b) % self.P
            y = tonelli_shanks(rhs, self.P)
            attempts += 1
            if y is not None:
                if self.verbose:
                    # print(f"[DEBUG] Found point on curve after {attempts} attempts:")
                    print(f"[DEBUG] G = ({x}, {y})")
                return (x, y)

    def point_add(self, P1, P2):
        if P1 is None:
            return P2
        if P2 is None:
            return P1

        x1, y1 = P1
        x2, y2 = P2

        if x1 == x2 and (y1 + y2) % self.P == 0:
            if self.verbose:
                print("[DEBUG] Point at infinity detected")
            return None  # point at infinity

        if P1 == P2:
            # doubling
            if y1 == 0:
                if self.verbose:
                    print("[DEBUG] Point doubling with y=0")
                return None
            lam = (3 * x1 * x1 + self.a) * pow(2 * y1, -1, self.P) % self.P
            # if self.verbose:
                # print(f"[DEBUG] Point doubling: λ = {lam}")
        else:
            # point addition
            lam = (y2 - y1) * pow(x2 - x1, -1, self.P) % self.P
            # if self.verbose:
                # print(f"[DEBUG] Point addition: λ = {lam}")

        x3 = (lam * lam - x1 - x2) % self.P
        y3 = (lam * (x1 - x3) - y1) % self.P
        # if self.verbose:
            # print(f"[DEBUG] Result point: ({x3}, {y3})")
        return (x3, y3)

    def scalar_mult(self, k, point):
        if self.verbose:
            print(f"[DEBUG] Scalar multiplication: {k} * {point}")
        result = None
        addend = point
        while k:
            if k & 1:
                result = self.point_add(result, addend)
            addend = self.point_add(addend, addend)
            k >>= 1
        if self.verbose:
            print(f"[DEBUG] Scalar multiplication result: {result}")
        return result

    def generate_keys(self):
        self.private_key = random.randrange(2, self.P)
        if self.verbose:
            print(f"[DEBUG] Generated private key: {self.private_key}")
        self.public_key = self.scalar_mult(self.private_key, self.G)
        if self.verbose:
            print(f"[DEBUG] Generated public key: {self.public_key}")
        return self.public_key

    def compute_shared_secret(self, other_public_key):
        if self.verbose:
            print(f"[DEBUG] Computing shared secret with other's public key: {other_public_key}")
        secret = self.scalar_mult(self.private_key, other_public_key)
        if self.verbose:
            print(f"[DEBUG] Computed shared secret: {secret}")
        return secret

    def setup(self):

        self.P = self.generate_prime()
        self.generate_curve_parameters()
        self.G = self.find_point_on_curve()

def measure_performance(key_size, trials=5, verbose=False):
    times = {
        'A': [],
        'B': [],
        'shared_key': []
    }
    
    for trial in range(trials):
        if verbose:
            print(f"\n[DEBUG] Starting trial {trial + 1}/{trials}")
        
        # setup
        ecc = ECDH(key_size, verbose)
        ecc.setup()
        
        # generate keys and measure time
        start = time.time()
        A_public = ecc.generate_keys()
        times['A'].append(time.time() - start)
        
        # create bob
        bob = ECDH(key_size, verbose)
        bob.P = ecc.P
        bob.a = ecc.a
        bob.b = ecc.b
        bob.G = ecc.G
        
        start = time.time()
        B_public = bob.generate_keys()
        times['B'].append(time.time() - start)
        
        #shared secret
        start = time.time()
        alice_secret = ecc.compute_shared_secret(B_public)
        bob_secret = bob.compute_shared_secret(A_public)
        times['shared_key'].append(time.time() - start)
        
        # verify match
        assert alice_secret == bob_secret, "Shared secrets don't match!"
        if verbose:
            print(f"[DEBUG] Trial {trial + 1} completed successfully")
    
    # averages
    avg_times = {
        'A': sum(times['A']) / trials,
        'B': sum(times['B']) / trials,
        'shared_key': sum(times['shared_key']) / trials
    }
    
    return avg_times

def main():
    # parse
    parser = argparse.ArgumentParser(description='ECDH Key Exchange Implementation')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # test different key size
    # key_sizes = [128, 192, 256]
    key_sizes = [128]
    
    print("\nPerformance Results (averaged over 5 trials):")
    print("\n       k | A Time (s) | B Time (s) | Shared Key Time (s)")
    print("-" * 55)
    
    for size in key_sizes:
        if args.verbose:
            print(f"\n[DEBUG] Testing key size: {size} bits")
        times = measure_performance(size, verbose=args.verbose)
        print(f"{size:8d} | {times['A']:10.6f} | {times['B']:10.6f} | {times['shared_key']:15.6f}")

if __name__ == "__main__":
    main() 