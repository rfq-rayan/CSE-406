/* Find the cache line size by running `getconf -a | grep CACHE` */
/* and for windows `wmic cpu get L2CacheSize` */
const LINESIZE = 64;

function readNlines(n) {
  /*
   * Implement this function to read n cache lines.
   * 1. Allocate a buffer of size n * LINESIZE.
   * 2. Read each cache line (read the buffer in steps of LINESIZE) 10 times.
   * 3. Collect total time taken in an array using `performance.now()`.
   * 4. Return the median of the time taken in milliseconds.
   */
  // new arrayBuffer of size n * LINESIZE
  const buffer = new ArrayBuffer(n * LINESIZE);
  // const view = new Uint8Array(buffer);
  const times = [];
  for (let i = 0; i < 10; i++) {
    const start = performance.now();
    // Read each cache line
    for (let j = 0; j < n; j++) {
      // Access the buffer in steps of LINESIZE
      const offset = j * LINESIZE;
      // Read the cache line (this is just a dummy read)
      // const line = view.slice(offset, offset + LINESIZE);
      const line = buffer[offset]; // This will trigger the read
    }
    const end = performance.now();
    times.push(end - start);
  }
  // Sort the times and return the median
  times.sort((a, b) => a - b);
  const medianIndex = Math.floor(times.length / 2);
  return times.length % 2 === 0
    ? (times[medianIndex - 1] + times[medianIndex]) / 2
    : times[medianIndex];
  
}

self.addEventListener("message", function (e) {
  if (e.data === "start") {
    const results = {};

    /* Call the readNlines function for n = 1, 10, ... 10,000,000 and store the result */
    for (let n = 1; n <= 10000000; n *= 10) {
      results[n] = readNlines(n);
    }
    self.postMessage(results);
  }
});
