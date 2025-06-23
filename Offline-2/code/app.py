from flask import Flask, send_from_directory, request, jsonify
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import time
# additional imports

app = Flask(__name__)

stored_traces = []
stored_heatmaps = []

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('static', path)

@app.route('/collect_trace', methods=['POST'])
def collect_trace():
    """ 
    Implement the collect_trace endpoint to receive trace data from the frontend and generate a heatmap.
    1. Receive trace data from the frontend as JSON
    2. Generate a heatmap using matplotlib
    3. Store the heatmap and trace data in the backend temporarily or save locally in static/heatmaps folder. 
    4. Return the heatmap image name/path and optionally other statistics to the frontend
    """
    try:
        # Get trace data from request
        if request.json is None:
            return jsonify({'error': 'Invalid JSON data'}), 400
            
        trace_data = request.json.get('traceData', [])
                
        if not trace_data:
            return jsonify({'error': 'No trace data provided'}), 400
        # Store the trace data
        stored_traces.append(trace_data)
        
        # Convert trace data to numpy array for heatmap
        trace_array = np.array(trace_data)
        
        # Create heatmap directory if it doesn't exist
        heatmap_dir = 'static/heatmaps'
        os.makedirs(heatmap_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        
        heatmap_filename = f'heatmap_{timestamp}.png'
        heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
        min_val = np.min(trace_array)
        max_val = np.max(trace_array)
        range_val = max_val - min_val
        samples = len(trace_array)
        
        # Generate heatmap for 1D sweep count data
        plt.figure(figsize=(12, 2))
        
        # Create a 2D representation by reshaping or repeating the data
        # For visualization, we'll create a heatmap by repeating the sweep counts
        # This makes it easier to see patterns over time
        trace_2d = trace_array.reshape(1, -1)  # Reshape to 2D for heatmap
        
        plt.imshow(trace_2d, cmap='hot', interpolation='nearest', aspect='auto')
        plt.title(f'Sweep Counts - Min: {min_val}, Max: {max_val}, Range: {range_val}, Samples: {samples}')
        # plt.xlabel('Time Interval')
        # plt.ylabel('Sweep Count')
        plt.axis('off')
        plt.colorbar(label='Sweep Count')
        plt.tight_layout()
        
        # Save heatmap
        plt.savefig(heatmap_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        # Store heatmap info
        heatmap_info = {
            'filename': heatmap_filename,
            'path': f'heatmaps/{heatmap_filename}',
            'timestamp': timestamp
        }
        stored_heatmaps.append(heatmap_info)
        
        return jsonify({
            'success': True,
            'heatmap': heatmap_info,
            'trace_count': len(stored_traces)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_results', methods=['POST'])
def clear_results():
    """ 
    Implment a clear results endpoint to reset stored data.
    1. Clear stored traces and heatmaps
    2. Return success/error message
    """
    try:
        global stored_traces, stored_heatmaps
        
        # Clear stored data
        stored_traces.clear()
        stored_heatmaps.clear()
        
        # Optionally, clean up heatmap files
        heatmap_dir = 'static/heatmaps'
        if os.path.exists(heatmap_dir):
            for filename in os.listdir(heatmap_dir):
                if filename.endswith('.png'):
                    os.remove(os.path.join(heatmap_dir, filename))
        
        return jsonify({
            'success': True,
            'message': 'All results cleared successfully'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_results', methods=['GET'])
def get_results():
    """
    Get all stored traces and heatmaps
    """
    try:
        return jsonify({
            'success': True,
            'traces': stored_traces,
            'heatmaps': stored_heatmaps
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Additional endpoints can be implemented here as needed.

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)