function app() {
  return {
    /* This is the main app object containing all the application state and methods. */
    // The following properties are used to store the state of the application

    // results of cache latency measurements
    latencyResults: null,
    // local collection of trace data
    traceData: [],
    // Local collection of heapmap images
    heatmaps: [],

    // Current status message
    status: "",
    // Is any worker running?
    isCollecting: false,
    // Is the status message an error?
    statusIsError: false,
    // Show trace data in the UI?
    showingTraces: false,

    // Collect latency data using warmup.js worker
    async collectLatencyData() {
      this.isCollecting = true;
      this.status = "Collecting latency data...";
      this.latencyResults = null;
      this.statusIsError = false;
      this.showingTraces = false;

      try {
        // Create a worker
        let worker = new Worker("warmup.js");

        // Start the measurement and wait for result
        const results = await new Promise((resolve) => {
          worker.onmessage = (e) => resolve(e.data);
          worker.postMessage("start");
        });

        // Update results
        this.latencyResults = results;
        this.status = "Latency data collection complete!";

        // Terminate worker
        worker.terminate();
      } catch (error) {
        console.error("Error collecting latency data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Collect trace data using worker.js and send to backend
    async collectTraceData() {
       /* 
        * Implement this function to collect trace data.
        * 1. Create a worker to run the sweep function.
        * 2. Collect the trace data from the worker.
        * 3. Send the trace data to the backend for temporary storage and heatmap generation.
        * 4. Fetch the heatmap from the backend and add it to the local collection.
        * 5. Handle errors and update the status.
        */

      this.isCollecting = true;
      this.status = "Collecting trace data...";
      this.traceData = [];
      this.statusIsError = false; 
      this.showingTraces = false;
      
      try {
        // Create a worker
        console.log('Creating worker...');
        let worker = new Worker("worker.js");

        // Start the trace collection
        this.status = "Running cache sweep...";
        console.log('Starting cache sweep...');
        const trace = await new Promise((resolve, reject) => {
          worker.onmessage = (e) => {
            console.log('Received data from worker:', e.data);
            if (e.data.error) {
              reject(new Error(e.data.error));
            } else if (e.data.data) {
              // Handle new format with data wrapper
              if (e.data.truncated) {
                console.warn(`Data was truncated from ${e.data.originalSize} to ${e.data.truncatedSize} intervals`);
                this.status = `Cache sweep complete (truncated: ${e.data.truncatedSize}/${e.data.originalSize} intervals)`;
              }
              resolve(e.data.data);
            } else {
              // Handle old format (backward compatibility)
              resolve(e.data);
            }
          };
          worker.onerror = (error) => {
            console.error('Worker error:', error);
            reject(error);
          };
          worker.postMessage("start");
        });
        
        console.log('Trace data received:', trace);
        
        // Update trace data locally
        this.traceData = trace;
        
        // Send trace data to backend for heatmap generation
        this.status = "Generating heatmap...";
        console.log('Sending trace data to backend...');
        const response = await fetch('/collect_trace', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ traceData: trace })
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        console.log('Backend response:', result);
        
        if (result.success) {
          // Add heatmap to local collection
          this.heatmaps.push(result.heatmap);
          this.status = `Trace data collection complete!`;
          this.showingTraces = true;
        } else {
          throw new Error(result.error || 'Failed to generate heatmap');
        }
        
        // Terminate worker
        worker.terminate();
        
      } catch (error) {
        console.error("Error collecting trace data:", error);
        this.status = `Error: ${error.message}`;
        this.statusIsError = true;
      } finally {
        this.isCollecting = false;
      }
    },

    // Download the trace data as JSON (array of arrays format for ML)
    async downloadTraces() {
       /* 
        * Implement this function to download the trace data.
        * 1. Fetch the latest data from the backend API.
        * 2. Create a download file with the trace data in JSON format.
        * 3. Handle errors and update the status.
        */
        
      try {
        this.status = "Fetching latest traces...";
        
        // Fetch latest data from backend
        const response = await fetch('/api/get_results');
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success && result.traces.length > 0) {
          // Create download data
          const downloadData = {
            traces: result.traces,
            metadata: {
              timestamp: new Date().toISOString(),
              trace_count: result.traces.length,
              heatmap_count: result.heatmaps.length
            }
          };
          
          // Create blob and download
          const blob = new Blob([JSON.stringify(downloadData, null, 2)], {
            type: 'application/json'
          });
          
          const url = URL.createObjectURL(blob);
          const a = document.createElement('a');
          a.href = url;
          a.download = `cache_traces_${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
          document.body.appendChild(a);
          a.click();
          document.body.removeChild(a);
          URL.revokeObjectURL(url);
          
          this.status = `Downloaded ${result.traces.length} traces successfully!`;
        } else {
          this.status = "No traces available to download";
        }
        
      } catch (error) {
        console.error("Error downloading traces:", error);
        this.status = `Error downloading traces: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Clear all results from the server
    async clearResults() {
      /* 
       * Implement this function to clear all results from the server.
       * 1. Send a request to the backend API to clear all results.
       * 2. Clear local copies of trace data and heatmaps.
       * 3. Handle errors and update the status.
       */
       
      try {
        this.status = "Clearing all results...";
        
        // Send clear request to backend
        const response = await fetch('/api/clear_results', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          }
        });
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        
        if (result.success) {
          // Clear local data
          this.traceData = [];
          this.heatmaps = [];
          this.latencyResults = null;
          this.showingTraces = false;
          this.status = "Cleared";
        } else {
          throw new Error(result.error || 'Failed to clear results');
        }
        
      } catch (error) {
        console.error("Error clearing results:", error);
        this.status = `Error clearing results: ${error.message}`;
        this.statusIsError = true;
      }
    },

    // Fetch existing results when page loads
    async fetchResults() {
      try {
        const response = await fetch('/api/get_results');
        
        if (response.ok) {
          const result = await response.json();
          
          if (result.success) {
            this.heatmaps = result.heatmaps || [];
            if (this.heatmaps.length > 0) {
              this.showingTraces = true;
            }
          }
        }
      } catch (error) {
        console.error("Error fetching existing results:", error);
      }
    },
  };
}
