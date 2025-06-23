import time
import json
import os
import signal
import sys
import random
import traceback
import socket
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import database
from database import Database

WEBSITES = [
    # websites of your choice
    "https://cse.buet.ac.bd/moodle/",
    "https://google.com",
    "https://prothomalo.com",
]

TRACES_PER_SITE = 10  # Start with 10 traces per site for testing
FINGERPRINTING_URL = "http://localhost:5000" 
OUTPUT_PATH = "dataset.json"

# Initialize the database to save trace data reliably
database.db = Database(WEBSITES)

""" Signal handler to ensure data is saved before quitting. """
def signal_handler(sig, frame):
    print("\nReceived termination signal. Exiting gracefully...")
    try:
        database.db.export_to_json(OUTPUT_PATH)
    except:
        pass
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


"""
Some helper functions to make your life easier.
"""

def is_server_running(host='127.0.0.1', port=5000):
    """Check if the Flask server is running."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    return result == 0

def start_flask_server():
    """Start the Flask server if it's not already running."""
    if is_server_running():
        print("✓ Flask server is already running")
        return True
    
    try:
        print("Starting Flask server...")
        # Start the server in the background
        process = subprocess.Popen([
            sys.executable, 'app.py'
        ], cwd=os.path.dirname(__file__) or '.')
        
        # Wait a few seconds for the server to start
        time.sleep(5)
        
        if is_server_running():
            print("✓ Flask server started successfully")
            return True
        else:
            print("✗ Flask server failed to start")
            return False
            
    except Exception as e:
        print(f"✗ Error starting Flask server: {e}")
        return False

def setup_webdriver():
    """Set up the Selenium WebDriver with Chrome options."""
    chrome_options = Options()
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--disable-web-security")
    chrome_options.add_argument("--allow-running-insecure-content")
    chrome_options.add_argument("--disable-extensions")
    # Uncomment the next line if you want to run headless (without GUI)
    # chrome_options.add_argument("--headless")
    
    # Try to use local ChromeDriver first, then fallback to webdriver-manager
    chromedriver_path = os.path.join(os.getcwd(), "chromedriver137", "chromedriver-win64", "chromedriver.exe")
    if os.path.exists(chromedriver_path):
        print(f"Using local ChromeDriver: {chromedriver_path}")
        service = Service(chromedriver_path)
    else:
        chromedriver_path = os.path.join(os.getcwd(), "chromedriver-win64", "chromedriver.exe")
        if os.path.exists(chromedriver_path):
            print(f"Using local ChromeDriver: {chromedriver_path}")
            service = Service(chromedriver_path)
        else:
            print("Using webdriver-manager to download ChromeDriver")
            service = Service(ChromeDriverManager().install())
    
    driver = webdriver.Chrome(service=service, options=chrome_options)
    
    # Execute script to prevent detection of automated software
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    return driver

def retrieve_traces_from_backend(driver):
    """Retrieve traces from the backend API."""
    traces = driver.execute_script("""
        return fetch('/api/get_results')
            .then(response => response.ok ? response.json() : {traces: []})
            .then(data => data.traces || [])
            .catch(() => []);
    """)
    
    count = len(traces) if traces else 0
    print(f"  - Retrieved {count} traces from backend API" if count else "  - No traces found in backend storage")
    return traces or []

def clear_trace_results(driver, wait):
    """Clear all results from the backend by pressing the button."""
    clear_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Clear all results')]")
    clear_button.click()

    wait.until(EC.text_to_be_present_in_element(
        (By.XPATH, "//div[@role='alert']"), "Cleared"))
    
def is_collection_complete():
    """Check if target number of traces have been collected."""
    current_counts = database.db.get_traces_collected()
    remaining_counts = {website: max(0, TRACES_PER_SITE - count) 
                      for website, count in current_counts.items()}
    return sum(remaining_counts.values()) == 0

"""
Your implementation starts here.
"""

def collect_single_trace(driver, wait, website_url):
    """ Implement the trace collection logic here. 
    1. Open the fingerprinting website
    2. Click the button to collect trace
    3. Open the target website in a new tab
    4. Interact with the target website (scroll, click, etc.)
    5. Return to the fingerprinting tab and close the target website tab
    6. Wait for the trace to be collected
    7. Return success or failure status
    """
    try:
        print(f"  - Starting trace collection for {website_url}")
        
        # Ensure we're on the fingerprinting page
        if driver.current_url != FINGERPRINTING_URL:
            driver.get(FINGERPRINTING_URL)
            time.sleep(2)
        
        # Find and click the "Collect Trace Data" button
        collect_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, "//button[contains(text(), 'Collect Trace Data')]")
        ))
        collect_button.click()
        print("  - Clicked 'Collect Trace Data' button")
        
        # Wait a moment for the trace collection to start
        time.sleep(1)
        
        # Open target website in a new tab
        driver.execute_script("window.open('');")
        driver.switch_to.window(driver.window_handles[1])
        driver.get(website_url)
        print(f"  - Opened {website_url} in new tab")
        
        # Wait for page to load
        time.sleep(3)
        
        # Simulate user activity - random scrolling
        scroll_actions = random.randint(5, 15)
        for i in range(scroll_actions):
            # Random scroll position
            scroll_position = random.randint(100, 1000)
            driver.execute_script(f"window.scrollTo(0, {scroll_position});")
            time.sleep(random.uniform(0.5, 2.0))
        
        print(f"  - Performed {scroll_actions} scroll actions")
        
        # Close the target website tab and return to fingerprinting tab
        driver.close()
        driver.switch_to.window(driver.window_handles[0])
        print("  - Returned to fingerprinting tab")
        
        # Wait for trace collection to complete (look for status change)
        try:
            # Wait for either success or error message
            wait.until(lambda driver: any([
                "Trace data collection complete!" in driver.find_element(By.XPATH, "//div[@role='alert']").text,
                "Error:" in driver.find_element(By.XPATH, "//div[@role='alert']").text
            ]))
            
            status_text = driver.find_element(By.XPATH, "//div[@role='alert']").text
            if "Trace data collection complete!" in status_text:
                print("  - Trace collection completed successfully")
                return True
            else:
                print(f"  - Trace collection failed: {status_text}")
                return False
                
        except Exception as e:
            print(f"  - Timeout waiting for trace completion: {e}")
            return False
        
    except Exception as e:
        print(f"  - Error during trace collection: {str(e)}")
        # Try to clean up - close any extra tabs
        try:
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
        except:
            pass
        return False

def collect_fingerprints(driver, target_counts=None):
    """ Implement the main logic to collect fingerprints.
    1. Calculate the number of traces remaining for each website
    2. Open the fingerprinting website
    3. Collect traces for each website until the target number is reached
    4. Save the traces to the database
    5. Return the total number of new traces collected
    """
    wait = WebDriverWait(driver, 20)
    total_collected = 0
    
    try:
        # Get current counts from database
        current_counts = database.db.get_traces_collected()
        
        if target_counts is None:
            target_counts = {website: TRACES_PER_SITE for website in WEBSITES}
        
        # Calculate remaining traces needed
        remaining_counts = {}
        for website in WEBSITES:
            current = current_counts.get(website, 0)
            target = target_counts.get(website, TRACES_PER_SITE)
            remaining_counts[website] = max(0, target - current)
        
        print(f"Current trace counts: {current_counts}")
        print(f"Remaining traces needed: {remaining_counts}")
        
        # Open fingerprinting website
        driver.get(FINGERPRINTING_URL)
        time.sleep(3)
        
        # Collect traces for each website
        for website in WEBSITES:
            website_url = website
            needed = remaining_counts[website]
            
            if needed <= 0:
                print(f"Skipping {website_url} - target already reached")
                continue
                
            print(f"\nCollecting {needed} traces for {website_url}")
            
            for i in range(needed):
                print(f"  Collecting trace {i+1}/{needed} for {website_url}")
                
                # Clear previous results before collecting new trace
                try:
                    clear_trace_results(driver, wait)
                    time.sleep(1)
                except Exception as e:
                    print(f"  - Warning: Could not clear results: {e}")
                
                # Collect single trace
                success = collect_single_trace(driver, wait, website_url)
                
                if success:
                    # Retrieve the trace data from backend
                    traces = retrieve_traces_from_backend(driver)
                    
                    if traces:
                        # Save the most recent trace to database
                        latest_trace = traces[-1]  # Get the last trace
                        site_index = current_counts.get(website, 0) + i + 1
                        
                        if database.db.save_trace(website, site_index, latest_trace):
                            total_collected += 1
                            print(f"  - Successfully saved trace {site_index} for {website}")
                        else:
                            print(f"  - Failed to save trace to database")
                    else:
                        print(f"  - No trace data retrieved from backend")
                else:
                    print(f"  - Failed to collect trace for {website_url}")
                
                # Small delay between traces
                time.sleep(2)
        
        print(f"\nCollection completed. Total new traces collected: {total_collected}")
        return total_collected
        
    except Exception as e:
        print(f"Error in collect_fingerprints: {str(e)}")
        traceback.print_exc()
        return total_collected

def main():
    """ Implement the main function to start the collection process.
    1. Check if the Flask server is running
    2. Initialize the database
    3. Set up the WebDriver
    4. Start the collection process, continuing until the target number of traces is reached
    5. Handle any exceptions and ensure the WebDriver is closed at the end
    6. Export the collected data to a JSON file
    7. Retry if the collection is not complete
    """
    print("Starting website fingerprinting data collection...")
    
    # Check if Flask server is running, try to start it if not
    if not is_server_running():
        print("Flask server is not running, attempting to start it...")
        if not start_flask_server():
            print("ERROR: Could not start Flask server automatically!")
            print("Please start the server manually with: python app.py")
            sys.exit(1)
    else:
        print("✓ Flask server is running")
    
    # Initialize database
    try:
        database.db.init_database()
        print("✓ Database initialized")
    except Exception as e:
        print(f"ERROR: Failed to initialize database: {e}")
        sys.exit(1)
    
    driver = None
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries and not is_collection_complete():
        try:
            retry_count += 1
            print(f"\n--- Collection Attempt {retry_count}/{max_retries} ---")
            
            # Set up WebDriver
            print("Setting up WebDriver...")
            driver = setup_webdriver()
            print("✓ WebDriver initialized")
            
            # Start collection process
            collected = collect_fingerprints(driver)
            print(f"✓ Collected {collected} new traces this session")
            
            # Check if we've reached our targets
            current_counts = database.db.get_traces_collected()
            print(f"Current trace counts: {current_counts}")
            
            if is_collection_complete():
                print("✓ Target number of traces reached for all websites!")
                break
            else:
                remaining = {website: max(0, TRACES_PER_SITE - current_counts.get(website, 0)) 
                           for website in WEBSITES}
                print(f"Still need: {remaining}")
                
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
            break
        except Exception as e:
            print(f"ERROR during collection attempt {retry_count}: {str(e)}")
            traceback.print_exc()
            
        finally:
            # Always close the driver
            if driver:
                try:
                    driver.quit()
                    print("✓ WebDriver closed")
                except:
                    pass
                driver = None
        
        # Wait before retry if not complete
        if retry_count < max_retries and not is_collection_complete():
            print(f"Waiting 10 seconds before retry {retry_count + 1}...")
            time.sleep(10)
    
    # Export final dataset
    try:
        print(f"\nExporting collected data to {OUTPUT_PATH}...")
        database.db.export_to_json(OUTPUT_PATH)
        print("✓ Data export completed")
        
        # Print final statistics
        final_counts = database.db.get_traces_collected()
        total_traces = sum(final_counts.values())
        print(f"\n--- Final Statistics ---")
        print(f"Total traces collected: {total_traces}")
        for website, count in final_counts.items():
            print(f"  {website}: {count}/{TRACES_PER_SITE}")
        
    except Exception as e:
        print(f"ERROR during data export: {e}")
    
    print("Collection process finished.")

if __name__ == "__main__":
    main()
