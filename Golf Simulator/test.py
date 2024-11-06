import socket
import json

# Server address and port
SERVER_ADDRESS = '192.168.174.166'  # Replace with the server's IP address
SERVER_PORT = 3333  # Replace with the server's port number

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the server
    print(f'Connecting to {SERVER_ADDRESS}:{SERVER_PORT}...')
    sock.connect((SERVER_ADDRESS, SERVER_PORT))
    print('Connected!')

    # Send a request for JSON data
    request = 'GET_JSON_DATA'
    print(f'Sending request: {request}')
    sock.sendall(request.encode())

    # Receive response from the server
    data = sock.recv(1024)
    response = data.decode()
    print(f'Received response: {response}')

    # Parse the JSON data
    try:
        json_data = json.loads(response)
        speed = json_data['speed']
        latitude = json_data['lat']
        longitude = json_data['long']
        heading = json_data['heading']

        print(f'Speed: {speed*3.6} km/h')
        print(f'Latitude: {latitude}')
        print(f'Longitude: {longitude}')
        print(f'Heading: {heading} degrees')

    except json.JSONDecodeError:
        print('Error: Invalid JSON data received from the server')

except Exception as e:
    print(f'Error: {e}')

finally:
    # Clean up
    print('Closing socket')
    sock.close()

SERVER_PORT = 4444  # Replace with the server's port number

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the server
    print(f'Connecting to {SERVER_ADDRESS}:{SERVER_PORT}...')
    sock.connect((SERVER_ADDRESS, SERVER_PORT))
    print('Connected!')

    # Send a request for JSON data
    request = 'GET_SENSOR_DATA'
    print(f'Sending request: {request}')
    sock.sendall(request.encode())

    # Receive response from the server
    data = sock.recv(1024)
    response = data.decode()
    print(f'Received response: {response}')

    # Parse the JSON data
    try:
        json_data = json.loads(response)
        temp = json_data['Temperature']
        humid = json_data['Humidity']

        print(f'Temperature: {temp} C degrees')
        print(f'Humidity: {humid}')

    except json.JSONDecodeError:
        print('Error: Invalid JSON data received from the server')

except Exception as e:
    print(f'Error: {e}')

finally:
    # Clean up
    print('Closing socket')
    sock.close()