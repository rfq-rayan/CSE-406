/* Find the cache line size by running `getconf -a | grep CACHE` */
const LINESIZE = 64;
/* Find the L3 size by running `getconf -a | grep CACHE` */
// Reduced cache size for memory efficiency
const LLCSIZE = 16 * 1024 * 1024; // 24MB
/* Time period for collecting traces in milliseconds */
const TIME = 10000;
/* Time period for each sweep in milliseconds */
const P = 10; 

function sweep(P) {
    /*
     * Implement this function to run a sweep of the cache.
     * 1. Allocate a buffer of size LLCSIZE.
     * 2. Read each cache line (read the buffer in steps of LINESIZE).
     * 3. Count the number of complete sweeps that can be performed in a time period of P milliseconds.
     * 4. Store the count in an array of size K, where K = TIME / P.
     * 5. Return the array of sweep counts.
     */
    
    // Allocate buffer of size LLCSIZE
    const buffer = new ArrayBuffer(LLCSIZE);
    // const view = new Uint8Array(buffer);
    
    // Calculate number of cache lines and time intervals
    const numCacheLines = Math.floor(LLCSIZE / LINESIZE);
    const numIntervals = Math.floor(TIME / P);
    
    console.log(`Cache sweep: ${numCacheLines} cache lines, ${numIntervals} intervals`);
    
    // Initialize result array to store sweep counts for each time interval
    const traces = [];
    
    // For each time interval
    for (let interval = 0; interval < numIntervals; interval++) {
        let sweepCount = 0;
        const startTime = performance.now();
        
        // Keep performing complete sweeps for P milliseconds
        while (performance.now() - startTime < P) {
            // Perform one complete sweep through all cache lines
            for (let line = 0; line < numCacheLines; line++) {
                const offset = line * LINESIZE;
                // Read the cache line (access the memory)
                const dummy = buffer[offset]; // This will trigger the read
            }
            sweepCount++;
        }
        
        traces.push(sweepCount);
        
        // Report progress for long-running operations
        if (interval % 100 === 0) {
            console.log(`Progress: ${interval + 1}/${numIntervals} intervals completed`);
        }
    }
    
    console.log(`Cache sweep completed: ${traces.length} intervals`);
    return traces;
}   

self.addEventListener('message', function(e) {
    /* Call the sweep function and return the result */
    console.log('Worker received message:', e.data);
    if (e.data === "start") {
        try {
            console.log('Starting cache sweep...');
            const traceData = sweep(P);
            console.log('Cache sweep completed, trace data length:', traceData.length);
            
            // Check data size before sending
            const dataSize = JSON.stringify(traceData).length;
            console.log('Data size (bytes):', dataSize);
            
            if (dataSize > 10 * 1024 * 1024) { // 10MB limit
                console.warn('Data too large, truncating...');
                // Send only first 100 intervals to prevent memory issues
                const truncatedData = traceData.slice(0, 100);
                self.postMessage({
                    data: truncatedData,
                    truncated: true,
                    originalSize: traceData.length,
                    truncatedSize: truncatedData.length
                });
            } else {
                self.postMessage({
                    data: traceData,
                    truncated: false,
                    size: traceData.length
                });
            }
        } catch (error) {
            console.error('Error in worker:', error);
            self.postMessage({ error: error.message });
        }
    }
});